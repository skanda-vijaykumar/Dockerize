from typing import List, Dict, Any, Optional, Type
from langchain.tools import BaseTool
from connector_dimension_tool import ConnectorDimensionTool
import re
import json

valid_families = ["CMM", "DMM", "AMM", "EMM", "DBM", "DFM"]

class ConnectorDimensionLangchainTool(BaseTool):
    name: str = "connector_dimensions"
    
    # "- Specific connector dimensions (e.g., "What are the dimensions of CMM with 20 pins?")
    # - Finding connectors that meet specific dimensions constraints (e.g., "Which connector fits 20 pins in less than 19x4mm?")
    # - Finding the smallest connector for a specific pin count (e.g., "Which connector has the least dimensions for 20 pins?")
    description: str ="""
ONLY use this tool when user needs information about PHYSICAL DIMENSIONS / SPACE and PIN COUNT of connectors (AMM, CMM, DMM, EMM):
    - Specific connector dimensions (e.g., "What are the dimensions of CMM with 20 pins?")
    - Finding connectors that meet specific dimensions constraints (e.g., "Which connector fits 20 pins in less than 19x4mm?")
    - Finding the smallest connector for a specific pin count (e.g., "Which connector has the least dimensions for 20 pins?","actually i need 12 contacts for data and 4 contacts with 20A")
    - Checking fitment of connectors and pins (e.g., "What is the maximum number of contacts to accommodate in a DMM?")
    - Questions about number of pins (e.g., "I need 100 pins DMM")
NEVER use this tool for:
    - Pitch size
    - Cable compatibility questions
    - Accessory matching
    - Part number lookups
    - Connector specifications other than dimensions
    - Any question containing terms like "compatible with", "goes with", or "works with"
    
Input should be a clear and complete sentence exactly like user's input, explicitly mention power and signal if mentioned in user input.
"""
    return_direct: bool = False
    
    # Define dimension_tool as a class attribute with default None
    dimension_tool: Optional[ConnectorDimensionTool] = None
    
    def __init__(self, data_dir: str):
        super().__init__()
        # Then create the dimension tool
        self.dimension_tool = ConnectorDimensionTool(data_dir)
    
    def extract_all_connector_families(self, query: str) -> List[str]:
        mentioned_families = []
        query_upper = query.upper()
        
        for family in valid_families:
            if family in query_upper:
                mentioned_families.append(family)
        
        return mentioned_families
        
    def get_connectors_for_all_families(self, families: List[str], pin_count: int) -> Dict[str, List[Dict]]:
        results = {}
        
        for family in families:
            # Use the existing method to find connectors for this family
            connectors = self.dimension_tool.find_by_series_and_pins(family, pin_count)
            if connectors:
                results[family] = connectors
                
        return results
    
    def find_extremal_connector(self, connectors_by_family: Dict[str, List[Dict]], criteria: str = "largest") -> Dict:
        all_connectors = []
        for family, connectors in connectors_by_family.items():
            all_connectors.extend(connectors)
        
        if not all_connectors:
            return None
        
        if criteria == "largest":
            return max(all_connectors, key=lambda x: x['length'] * x['height'])
        elif criteria == "smallest":
            return min(all_connectors, key=lambda x: x['length'] * x['height'])
        else:
            return None
            
    def format_multi_family_response(self, connectors_by_family: Dict[str, List[Dict]], 
                                    pin_count: int, criteria: str = None) -> str:
        if not connectors_by_family:
            return f"No connectors found with {pin_count} pins in any of the specified families."
        
        response_parts = []
        
        # Add summary for each family
        for family, connectors in connectors_by_family.items():
            if connectors:
                family_response = f"\n\n{family} connectors with {pin_count} pins:"
                
                # Sort connectors by area for better comparison
                sorted_connectors = sorted(connectors, key=lambda x: x['length'] * x['height'])
                
                for conn in sorted_connectors:
                    area = conn['length'] * conn['height']
                    family_response += f"\n- {family}{conn['series']} {conn['gender']}: {conn['dimensions']} (Area: {round(area, 2)} mm²)"
                
                response_parts.append(family_response)
            else:
                response_parts.append(f"\n\nNo {family} connectors found with {pin_count} pins.")
        
        # If criteria specified, add the extremal connector
        if criteria:
            extremal_connector = self.find_extremal_connector(connectors_by_family, criteria)
            if extremal_connector:
                area = extremal_connector['length'] * extremal_connector['height']
                family = extremal_connector['connector_family']
                series = extremal_connector['series']
                gender = extremal_connector['gender']
                
                response_parts.insert(0, f"The connector with the {criteria} dimensions among all families is "
                                      f"{family}{series} {gender}: {extremal_connector['dimensions']} "
                                      f"(Area: {round(area, 2)} mm²)")
        
        return "\n".join(response_parts)

    def handle_multi_family_query(self, query: str, pin_count: int, criteria: str = None) -> str:
        families = self.extract_all_connector_families(query)
        
        if not families:
            return "No specific connector families identified in the query."
            
        connectors_by_family = self.get_connectors_for_all_families(families, pin_count)
        
        return self.format_multi_family_response(connectors_by_family, pin_count, criteria)
    
    def _run(self, query: str) -> str:
        if self.dimension_tool is None:
            return "Error: Connector dimension tool not initialized properly."
        
        query_lower = query.lower()
        
        # Check for comparative queries that need special handling
        comparative_query = False
        criteria = None
        
        if any(word in query_lower for word in ["largest", "biggest", "maximum", "widest", "tallest"]):
            comparative_query = True
            criteria = "largest"
        elif any(word in query_lower for word in ["smallest", "tiniest", "minimum", "compact", "narrowest"]):
            comparative_query = True
            criteria = "smallest"
            
        # Process signal and power contacts if mentioned
        signal_pattern = r'(\d+)\s*(?:signal|sig)(?:\s*contacts?|\s*pins?)'
        power_pattern = r'(\d+)\s*(?:power|pwr)(?:\s*contacts?|\s*pins?)'
        
        signal_match = re.search(signal_pattern, query_lower)
        power_match = re.search(power_pattern, query_lower)
        
        total_contacts = None
        if signal_match and power_match:
            signal_contacts = int(signal_match.group(1))
            power_contacts = int(power_match.group(1))
            total_contacts = signal_contacts + (power_contacts * 4)
            print(f"Converting {signal_contacts} signal + {power_contacts} power contacts to {total_contacts} total contacts")
        
        # Extract pin count
        pin_count = None
        # Try standard pattern first
        pin_pattern = r'(\d+)\s*(?:pin|pins|contact|contacts)'
        pin_match = re.search(pin_pattern, query_lower)

        # If no match, try with qualifier pattern
        if not pin_match:
            pin_pattern_with_qualifier = r'(\d+)\s+(?:[a-z]+\s+)+(?:pin|pins|contact|contacts)'
            pin_match = re.search(pin_pattern_with_qualifier, query_lower)

        if pin_match:
            pin_count = int(pin_match.group(1))
        elif total_contacts:
            pin_count = total_contacts
        
        # Extract all mentioned families
        families = self.extract_all_connector_families(query)
        
        # Check if query mentions multiple families
        if len(families) > 1 and pin_count:
            if "among" in query_lower or "between" in query_lower or comparative_query:
                return self.handle_multi_family_query(query, pin_count, criteria)
            else:
                # Process each family separately
                all_results = []
                for family in families:
                    # Create a focused query for this family
                    family_query = f"What are the dimensions of {family} with {pin_count} pins?"
                    result = self.dimension_tool.generate_response(family_query)
                    all_results.append(f"Results for {family}:\n{result}")
                
                return "\n\n" + "\n\n".join(all_results)
        
        # For single family or generic queries, use the default processor
        return self.dimension_tool.generate_response(query)
    
    async def _arun(self, query: str) -> str:
        return self._run(query)