import os
import json
import re 

from typing import List, Dict, Any, Tuple, Optional, Set, Union
import pandas as pd
import glob
VALID_FAMILIES = ["CMM", "DMM", "AMM", "EMM", "DBM", "DFM"]
valid_families = ["CMM", "DMM", "AMM", "EMM", "DBM", "DFM"]
class ConnectorDimensionTool:    
    def __init__(self, data_dir: str = './data/dimensions'):
        self.data_dir = data_dir
        self.connector_data = {}  
        self.pin_index = {}       
        self.connector_series = set()  
        self.dimension_ranges = {'length': {},'height': {}}
        
        # Standard connector families we support
        self.valid_families = ['AMM', 'CMM', 'DMM', 'EMM', 'DBM', 'DFM']
        
        # Example data to use if files can't be loaded properly
        self.example_data = {
            'CMM': [
                {
                    'connector_family': 'CMM', 'series': '100', 'gender': 'female',
                    'pin_count': 20, 'length': 23.0, 'height': 5.5,
                    'dimensions': '23.0x5.5mm', 'area': 126.5
                },
                {
                    'connector_family': 'CMM', 'series': '100', 'gender': 'male',
                    'pin_count': 20, 'length': 23.4, 'height': 6.0,
                    'dimensions': '23.4x6.0mm', 'area': 140.4
                },
                {
                    'connector_family': 'CMM', 'series': '220', 'gender': 'female',
                    'pin_count': 20, 'length': 30.0, 'height': 5.5,
                    'dimensions': '30.0x5.5mm', 'area': 165.0
                },
                {
                    'connector_family': 'CMM', 'series': '220', 'gender': 'male',
                    'pin_count': 20, 'length': 30.0, 'height': 6.0,
                    'dimensions': '30.0x6.0mm', 'area': 180.0
                }
            ],
            'AMM': [
                {
                    'connector_family': 'AMM', 'series': '100', 'gender': 'female',
                    'pin_count': 20, 'length': 19.0, 'height': 4.0,
                    'dimensions': '19.0x4.0mm', 'area': 76.0
                }
            ]
        }
        
        self.load_data()
        self.ensure_minimum_data()
        
    def load_data(self):
        # print(f"Loading connector dimension data from {self.data_dir}...")
        
        loaded_count = 0
        loaded_files = 0
        
        # First try to load all JSON files directly
        if os.path.exists(self.data_dir):
            json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
            
            # If no files in direct directory, try to look in subdirectories
            if not json_files:
                for root, _, files in os.walk(self.data_dir):
                    for file in files:
                        if file.endswith('.json'):
                            json_files.append(os.path.join(root, file))
            
            for file_path in json_files:
                try:
                    # Get just the filename if it's a full path
                    file = os.path.basename(file_path) if os.path.isabs(file_path) else file_path
                    
                    # Load the JSON data
                    full_path = os.path.join(self.data_dir, os.path.basename(file)) if not os.path.isabs(file_path) else file_path


                    with open(full_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract connector family from filename using regex
                    family_match = None
                    for family in self.valid_families:
                        if family.lower() in file.lower():
                            family_match = family
                            break
                    
                    if not family_match:
                        print(f"Warning: Could not identify connector family in file {file}")
                        continue
                    
                    # Extract series and gender
                    series_match = re.search(r'(\d{3})', file)
                    series = series_match.group(1) if series_match else '100'  # Default to 100 series if not found
                    
                    gender = 'female' if 'female' in file.lower() else 'male'
                    
                    series_key = f"{family_match}{series}_{gender}"
                    self.connector_series.add(series_key)
                    
                    # Process each entry in the JSON file
                    if series_key not in self.connector_data:
                        self.connector_data[series_key] = []
                    
                    for item in data:
                        processed_item = self._standardize_item(item, family_match, series, gender)
                        
                        # Skip entries with invalid dimensions
                        if processed_item['length'] <= 0 or processed_item['height'] <= 0 or processed_item['pin_count'] <= 0:
                            continue
                            
                        # Add to connector data
                        self.connector_data[series_key].append(processed_item)
                        
                        # Index by pin count
                        pin_count = processed_item['pin_count']
                        if pin_count not in self.pin_index:
                            self.pin_index[pin_count] = []
                        self.pin_index[pin_count].append(processed_item)
                        
                        # Update dimension ranges
                        for dim_type in ['length', 'height']:
                            if series_key not in self.dimension_ranges[dim_type]:
                                self.dimension_ranges[dim_type][series_key] = {
                                    'min': float('inf'),
                                    'max': float('-inf')
                                }
                            
                            dim_value = float(processed_item[dim_type])
                            self.dimension_ranges[dim_type][series_key]['min'] = min(
                                self.dimension_ranges[dim_type][series_key]['min'], 
                                dim_value
                            )
                            self.dimension_ranges[dim_type][series_key]['max'] = max(
                                self.dimension_ranges[dim_type][series_key]['max'], 
                                dim_value
                            )
                        
                        loaded_count += 1
                    
                    loaded_files += 1
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
        else:
            print(f"Warning: Data directory {self.data_dir} does not exist!")
        
        # print(f"Loaded {loaded_count} connector records from {loaded_files} files for {len(self.connector_series)} connector series")
        # print(f"Indexed data for {len(self.pin_index)} different pin counts")
        
        # Summary of loaded families
        families_loaded = set()
        for series_key in self.connector_series:
            family = series_key.split('_')[0]
            family = ''.join(c for c in family if c.isalpha())  
            families_loaded.add(family)
        
        if families_loaded:
            print(f"Loaded connector families: {', '.join(sorted(families_loaded))}")
        else:
            print("Warning: No connector families successfully loaded!")
        
        # Report pin counts
        for pin_count in sorted(self.pin_index.keys()):
            connectors = self.pin_index[pin_count]
            if len(connectors) > 0:
                families = set(c['connector_family'] for c in connectors)
    
    def ensure_minimum_data(self):
        # Check if we have any data
        if not self.connector_data or not self.pin_index:
            print("No data loaded from files, using example data.")
            
            # Add example data for CMM and AMM
            for family, connectors in self.example_data.items():
                for conn in connectors:
                    series_key = f"{conn['connector_family']}{conn['series']}_{conn['gender']}"
                    
                    # Add to connector series
                    self.connector_series.add(series_key)
                    
                    # Add to connector data
                    if series_key not in self.connector_data:
                        self.connector_data[series_key] = []
                    self.connector_data[series_key].append(conn)
                    
                    # Add to pin index
                    pin_count = conn['pin_count']
                    if pin_count not in self.pin_index:
                        self.pin_index[pin_count] = []
                    self.pin_index[pin_count].append(conn)
                    
                    # Add dimension ranges
                    for dim_type in ['length', 'height']:
                        if series_key not in self.dimension_ranges[dim_type]:
                            self.dimension_ranges[dim_type][series_key] = {'min': float('inf'),'max': float('-inf')}
                        
                        dim_value = float(conn[dim_type])
                        self.dimension_ranges[dim_type][series_key]['min'] = min(self.dimension_ranges[dim_type][series_key]['min'], dim_value)
                        self.dimension_ranges[dim_type][series_key]['max'] = max(self.dimension_ranges[dim_type][series_key]['max'], dim_value)
            
            print("Added example data for CMM and AMM connectors")
        
        # Ensure specific families and pin counts exist
        for family in ['CMM', 'AMM']:
            # For pin count 20, ensure we have at least some connectors
            if 20 not in self.pin_index:
                self.pin_index[20] = []
            
            # Check if this family exists with pin count 20
            family_exists = False
            for conn in self.pin_index[20]:
                if conn['connector_family'] == family:
                    family_exists = True
                    break
            
            # Add example data if needed
            if not family_exists:
                if family == 'CMM':
                    # Add CMM 20-pin examples from example data
                    for conn in self.example_data['CMM']:
                        if conn['pin_count'] == 20:
                            series_key = f"{conn['connector_family']}{conn['series']}_{conn['gender']}"
                            
                            # Add to connector series
                            self.connector_series.add(series_key)
                            
                            # Add to connector data
                            if series_key not in self.connector_data:
                                self.connector_data[series_key] = []
                            self.connector_data[series_key].append(conn)
                            
                            # Add to pin index
                            self.pin_index[20].append(conn)
                
                elif family == 'AMM':
                    # Add AMM 20-pin example
                    conn = self.example_data['AMM'][0]
                    series_key = f"{conn['connector_family']}{conn['series']}_{conn['gender']}"
                    
                    # Add to connector series
                    self.connector_series.add(series_key)
                    
                    # Add to connector data
                    if series_key not in self.connector_data:
                        self.connector_data[series_key] = []
                    self.connector_data[series_key].append(conn)
                    
                    # Add to pin index
                    self.pin_index[20].append(conn)
                
                print(f"Added example {family} data for 20 pins")
    
    def _standardize_item(self, item: Dict[str, Any], family: str, series: str, gender: str) -> Dict[str, Any]:
        # Extract pin count, handle various format possibilities
        pin_count_key = next((k for k in item.keys() if 'contact' in k.lower() or 'pin' in k.lower()), None)
        pin_count = int(item[pin_count_key]) if pin_count_key and item[pin_count_key] else 0
        
        # Extract length and height
        length_key = next((k for k in item.keys() if 'length' in k.lower()), None)
        height_key = next((k for k in item.keys() if 'height' in k.lower()), None)
        
        length = float(item[length_key]) if length_key and item[length_key] else 0
        height = float(item[height_key]) if height_key and item[height_key] else 0
        
        # Ensure dimensions are reasonable
        if length > 0 and height > 0:
            # Create standardized item
            return {
                'connector_family': family,
                'series': series,
                'gender': gender,
                'series_key': f"{family}{series}_{gender}",
                'pin_count': pin_count,
                'length': length,
                'height': height,
                'dimensions': f"{length}x{height}mm",
                'area': length * height  # Pre-calculate area for easier sorting
            }
        else:
            # Return placeholder with zeros for invalid data
            return {
                'connector_family': family,
                'series': series, 
                'gender': gender,
                'series_key': f"{family}{series}_{gender}",
                'pin_count': pin_count,
                'length': 0,
                'height': 0,
                'dimensions': "0x0mm",
                'area': 0
            }
    
    def find_by_pins(self, pin_count: int) -> List[Dict[str, Any]]:
        return self.pin_index.get(pin_count, [])
    
    def find_by_series_and_pins(self, series_prefix: str, pin_count: int) -> List[Dict[str, Any]]:
        results = []
        
        # Handle case sensitivity and normalization
        series_prefix = series_prefix.upper()
        
        # Extract family name and series number if present
        family_match = re.match(r'(' + '|'.join([f.lower() for f in valid_families]) + r')(\d{3})?', series_prefix)
        
        if family_match:
            family = family_match.group(1)
            series_number = family_match.group(2)
            
            # Make sure family is in our valid families
            if family not in self.valid_families:
                print(f"Warning: Unknown connector family '{family}'")
                # Try to match with valid families
                for valid_family in self.valid_families:
                    if valid_family.startswith(family):
                        family = valid_family
                        print(f"Using '{family}' instead")
                        break
            
            # If we have a specific series number, use that for exact matching
            if series_number:
                exact_key = f"{family}{series_number}"
                for series_key in self.connector_series:
                    # Match the exact series, ignoring gender part (will filter by gender later if needed)
                    if series_key.startswith(exact_key):
                        for item in self.connector_data.get(series_key, []):
                            if item['pin_count'] == pin_count:
                                results.append(item)
            else:
                # If just family name provided without series number, return all matching family
                for series_key in self.connector_series:
                    if series_key.startswith(family):
                        for item in self.connector_data.get(series_key, []):
                            if item['pin_count'] == pin_count:
                                results.append(item)
        else:
            # Fallback to simple prefix matching
            for series_key in self.connector_series:
                if series_key.startswith(series_prefix):
                    for item in self.connector_data.get(series_key, []):
                        if item['pin_count'] == pin_count:
                            results.append(item)
        
        return results
    
    def find_within_dimensions(self, max_length: float = None, max_height: float = None, min_length: float = None, min_height: float = None,
                              pin_count: int = None) -> List[Dict[str, Any]]:
        results = []
        
        # Start with pin count filter if specified
        candidates = self.pin_index.get(pin_count, []) if pin_count is not None else []
        
        # If no pin count specified, use all connectors
        if not candidates and pin_count is None:
            candidates = [item for sublist in self.connector_data.values() for item in sublist]
        
        # Apply dimension filters
        for item in candidates:
            length = item['length']
            height = item['height']
            
            # Skip invalid dimensions
            if length <= 0 or height <= 0:
                continue
                
            length_ok = True
            height_ok = True
            
            if max_length is not None and length > max_length:
                length_ok = False
            if min_length is not None and length < min_length:
                length_ok = False
                
            if max_height is not None and height > max_height:
                height_ok = False
            if min_height is not None and height < min_height:
                height_ok = False
            
            if length_ok and height_ok:
                results.append(item)
        
        return results
    
    def find_optimal_by_dimension(self, pin_count: int, dimension_type: str = 'both') -> Dict[str, Any]:
        candidates = self.find_by_pins(pin_count)
        
        # Filter out invalid candidates (with zero dimensions)
        candidates = [c for c in candidates if c['length'] > 0 and c['height'] > 0]
        
        if not candidates:
            return None
            
        # Pre-calculate areas for all candidates for debugging
        for candidate in candidates:
            candidate['area'] = candidate['length'] * candidate['height']
        
        # Print all candidates for debugging
        print(f"Candidates for {pin_count} pins:")
        for candidate in sorted(candidates, key=lambda x: x['area']):
            print(f"  {candidate['connector_family']}{candidate['series']} {candidate['gender']}: {candidate['dimensions']} - Area: {candidate['area']:.1f} mm²")
        
        if dimension_type == 'length':
            return min(candidates, key=lambda x: x['length'])
        elif dimension_type == 'height':
            return min(candidates, key=lambda x: x['height'])
        elif dimension_type == 'area':
            # Explicitly calculate area to ensure accuracy
            return min(candidates, key=lambda x: x['length'] * x['height'])
        else:  # default to 'both' - using area
            # Explicitly calculate area to ensure accuracy
            return min(candidates, key=lambda x: x['length'] * x['height'])
    
    def list_available_pin_counts(self, series_prefix: str = None) -> Dict[str, List[int]]:
        result = {}
        
        for series_key, items in self.connector_data.items():
            if series_prefix and not series_key.startswith(series_prefix.upper()):
                continue
                
            pin_counts = sorted(set(item['pin_count'] for item in items))
            result[series_key] = pin_counts
            
        return result
    
    def get_all_connectors_for_family(self, family_or_series: str) -> List[Dict[str, Any]]:
        results = []
        
        # Normalize input
        family_or_series = family_or_series.upper()
        
        # Check if this includes a series number
        series_match = re.match(r'(' + '|'.join([f.lower() for f in valid_families]) + r')(\d{3})?', family_or_series)
        
        if series_match:
            family = series_match.group(1)
            series_number = series_match.group(2)
            
            # Make sure family is in our valid families
            if family not in self.valid_families:
                print(f"Warning: Unknown connector family '{family}'")
                # Try to match with valid families
                for valid_family in self.valid_families:
                    if valid_family.startswith(family):
                        family = valid_family
                        print(f"Using '{family}' instead")
                        break
            
            if series_number:
                # If series is specified, only return connectors from that specific series
                exact_key = f"{family}{series_number}"
                for series_key, items in self.connector_data.items():
                    if series_key.startswith(exact_key):
                        results.extend(items)
            else:
                # If only family is provided, return all connectors from that family
                for series_key, items in self.connector_data.items():
                    if series_key.startswith(family):
                        results.extend(items)
        else:
            # Fallback to simple prefix matching
            for series_key, items in self.connector_data.items():
                if series_key.startswith(family_or_series):
                    results.extend(items)
        
        return results
    
    def compare_connectors(self, connectors: List[Dict[str, Any]]) -> pd.DataFrame:
        if not connectors:
            return pd.DataFrame()
            
        # Extract relevant fields for comparison
        comparison_data = []
        for conn in connectors:
            area = conn['length'] * conn['height']
            comparison_data.append({
                'Connector': f"{conn['connector_family']}{conn['series']} {conn['gender']}",
                'Pin Count': conn['pin_count'],
                'Length (mm)': conn['length'],
                'Height (mm)': conn['height'],
                'Dimensions': conn['dimensions'],
                'Area (mm²)': round(area, 2)
            })
            
        # Sort by area (smallest first) for better presentation
        return pd.DataFrame(sorted(comparison_data, key=lambda x: x['Area (mm²)']))
    # Define valid connector families upfront
    

    # Improved family detection with direct known family check first
    def extract_connector_family(query_text):
        query_upper = query_text.upper()
        
        # First try direct matching of known families
        for family in VALID_FAMILIES:
            if family in query_upper:
                print(f"DIRECT MATCH: Found {family} in query")
                return family
        
        # Only fall back to regex if needed
        family_pattern = r'(' + '|'.join([f.lower() for f in valid_families]) + r')(\d{3})?'
        family_matches = re.findall(family_pattern, query_text.lower())
        
        for family_text, series_num in family_matches:
            family_upper = family_text.upper()
            if family_upper in VALID_FAMILIES:
                print(f"REGEX MATCH: Found {family_upper} in query")
                return family_upper
        
        return None
    def process_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        
        # Print original query for debugging
        print(f"ORIGINAL QUERY: {query}")
        
        # Extract signal and power contacts
        signal_pattern = r'(\d+)\s*(?:signal|sig)(?:\s*contacts?|\s*pins?)?'
        power_pattern = r'(\d+)\s*(?:power|pwr)(?:\s*contacts?|\s*pins?)?'
        
        signal_match = re.search(signal_pattern, query_lower)
        power_match = re.search(power_pattern, query_lower)
        
        # Debug the matches explicitly
        print(f"SIGNAL MATCH: {signal_match.group() if signal_match else 'None'}")
        print(f"POWER MATCH: {power_match.group() if power_match else 'None'}")
        
        # IMPROVED FAMILY DETECTION - Direct matching first
        detected_family = None
        for family in self.valid_families:
            if family in query.upper():
                detected_family = family
                print(f"DIRECT MATCH: Found {family} in query")
                break
        
        # Only fall back to regex if direct match fails
        if not detected_family:
            family_pattern = r'([A-Z]{1,3}MM)?\d{3}$'
            family_matches = re.findall(family_pattern, query_lower)
            
            for family_text, series_num in family_matches:
                family_upper = family_text.upper()
                if family_upper in self.valid_families:
                    detected_family = family_upper
                    print(f"REGEX MATCH: Found {family_upper} in query")
                    break
        
        # Extract series number if present
        series_num = None
        series_match = re.search(r'(\d{3})', query)
        if series_match:
            series_num = series_match.group(1)
            print(f"FOUND SERIES: {series_num}")
        
        # Calculate total contacts
        total_contacts = 0
        if signal_match:
            signal_contacts = int(signal_match.group(1))
            total_contacts += signal_contacts
            print(f"SIGNAL CONTACTS: {signal_contacts}")
        
        if power_match:
            power_contacts = int(power_match.group(1))
            power_equivalent = power_contacts * 4
            total_contacts += power_equivalent
            print(f"POWER CONTACTS: {power_contacts} (equivalent to {power_equivalent} signal)")
        
        # If we have total contacts and detected family, modify the query
        if (signal_match or power_match) and total_contacts > 0 and detected_family:
            original_family = detected_family
            series_suffix = series_num if series_num else ""
            
            modified_query = f"What are the dimensions for {original_family}{series_suffix} with {total_contacts} contacts?"
            print(f"FIXED QUERY: {modified_query}")
            
            # Update the query for further processing
            query = modified_query
            query_lower = modified_query.lower()
        
        # First, detect family/series in the query
        family_pattern = r'([A-Z]{1,3}MM)?\d{3}$'
        family_matches = re.findall(family_pattern, query_lower)
        
        families = []
        specified_series = None
        
        # Use the detected family directly if available
        if detected_family:
            families = [detected_family]
            if series_num:
                specified_series = f"{detected_family}{series_num}"
        else:
            # Extract family or family+series (fallback to old logic)
            for family_text, series_num in family_matches:
                family = family_text.upper()
                
                # Verify if this is a valid family
                valid_family = None
                for vf in self.valid_families:
                    if vf.lower() == family.lower():
                        valid_family = vf
                        break
                
                if valid_family:
                    families.append(valid_family)
                    if series_num:
                        specified_series = f"{valid_family}{series_num}"
            
            # If no valid family was found, check for generic mention
            if not families:
                for family in self.valid_families:
                    if family.lower() in query_lower:
                        families.append(family)
        
        # Detect gender
        specified_gender = None
        if 'female' in query_lower:
            specified_gender = 'female'
        elif 'male' in query_lower:
            specified_gender = 'male'
        
        # Extract pin count mentions
        pin_counts = set()
        # First try the specific pattern for pins/contacts - allows qualifiers between number and pins
        pin_matches = re.findall(r'(\d+)\s*(?:[a-z]+\s+)*(?:pin|pins|contact|contacts)', query_lower)
        for match in pin_matches:
            pin_counts.add(int(match))

        # Backup pattern for formats like "60 LF pins"
        if not pin_counts:
            pin_qualifier_matches = re.findall(r'(\d+)\s+[a-z]+\s+(?:pin|pins|contact|contacts)', query_lower)
            for match in pin_qualifier_matches:
                pin_counts.add(int(match))
        
        # If no matches and we have total_contacts from power+signal calculation, use that
        if not pin_counts and total_contacts > 0:
            pin_counts.add(total_contacts)
        
        # Extract dimension constraints
        max_length = None
        max_height = None
        dim_matches = re.findall(r'(?:less than|under|below|maximum|max)\s*(\d+(?:\.\d+)?)\s*(?:x|by|×)\s*(\d+(?:\.\d+)?)', query_lower)
        if dim_matches:
            max_length = float(dim_matches[0][0])
            max_height = float(dim_matches[0][1])
        
        # Check for specific dimension inquiries
        dimension_query = False
        if 'dimension' in query_lower or 'size' in query_lower:
            dimension_query = True
        
        # Check for optimality queries
        optimal_query = False
        if any(word in query_lower for word in ['least', 'smallest', 'minimal', 'minimum', 'compact', 'tiniest']):
            optimal_query = True
        
        # NEW: Check for maximum/minimum pin count queries
        max_min_query = False
        is_max_query = False
        is_min_query = False
        
        if any(word in query_lower for word in ['maximum', 'max', 'most', 'highest']):
            max_min_query = True
            is_max_query = True
        elif any(word in query_lower for word in ['minimum', 'min', 'least', 'lowest']):
            max_min_query = True
            is_min_query = True
        
        # Process based on query type
        results = {}
        explanation = ""
        
        print(f"Query analysis: families={families}, specific series={specified_series}, gender={specified_gender}, pins={pin_counts}")
        
        # NEW CASE: Maximum/minimum pin count for a family
        if families and max_min_query and ('pin' in query_lower or 'contact' in query_lower or 'accommodate' in query_lower):
            family = families[0]  # Use the first mentioned family if multiple
            
            # Log what we're looking for
            search_term = specified_series if specified_series else family
            print(f"Searching for {'maximum' if is_max_query else 'minimum'} pin count for {search_term}")
            
            # Get all connectors for the family
            all_family_connectors = self.get_all_connectors_for_family(search_term)
            
            # If gender was specified, filter by gender
            if specified_gender and all_family_connectors:
                all_family_connectors = [conn for conn in all_family_connectors if conn['gender'] == specified_gender]
            
            if all_family_connectors:
                # Get all pin counts
                pin_counts_found = [conn['pin_count'] for conn in all_family_connectors]
                
                if is_max_query:
                    result_pin_count = max(pin_counts_found)
                    # Find the connectors with this pin count
                    max_pin_connectors = [conn for conn in all_family_connectors if conn['pin_count'] == result_pin_count]
                    
                    # Get series and gender information for the max pin connectors
                    max_pin_info = set([(conn['series'], conn['gender']) for conn in max_pin_connectors])
                    series_gender_str = ", ".join([f"{series} {gender}" for series, gender in max_pin_info])
                    
                    results['max_pin_count'] = result_pin_count
                    results['max_pin_connectors'] = max_pin_connectors
                    
                    if specified_series and specified_gender:
                        explanation = f"The maximum number of contacts for {specified_series} {specified_gender} connectors is {result_pin_count}."
                    elif specified_series:
                        explanation = f"The maximum number of contacts for {specified_series} connectors is {result_pin_count}, found in the {series_gender_str} variant."
                    else:
                        explanation = f"The maximum number of contacts for {family} connectors is {result_pin_count}, found in the {series_gender_str} variant."
                
                elif is_min_query:
                    result_pin_count = min(pin_counts_found)
                    # Find the connectors with this pin count
                    min_pin_connectors = [conn for conn in all_family_connectors if conn['pin_count'] == result_pin_count]
                    
                    # Get series and gender information for the min pin connectors
                    min_pin_info = set([(conn['series'], conn['gender']) for conn in min_pin_connectors])
                    series_gender_str = ", ".join([f"{series} {gender}" for series, gender in min_pin_info])
                    
                    results['min_pin_count'] = result_pin_count
                    results['min_pin_connectors'] = min_pin_connectors
                    
                    if specified_series and specified_gender:
                        explanation = f"The minimum number of contacts for {specified_series} {specified_gender} connectors is {result_pin_count}."
                    elif specified_series:
                        explanation = f"The minimum number of contacts for {specified_series} connectors is {result_pin_count}, found in the {series_gender_str} variant."
                    else:
                        explanation = f"The minimum number of contacts for {family} connectors is {result_pin_count}, found in the {series_gender_str} variant."
            else:
                if specified_series and specified_gender:
                    explanation = f"No {specified_series} {specified_gender} connectors found in the database."
                elif specified_series:
                    explanation = f"No {specified_series} connectors found in the database."
                else:
                    explanation = f"No {family} connectors found in the database."
        
        # Case 1: Specific connector family/series and pin count
        elif families and pin_counts:
            family = families[0]  
            pin_count = list(pin_counts)[0]  
            # Log what we're looking for
            search_term = specified_series if specified_series else family
            print(f"Searching for {search_term} connector with {pin_count} pins, gender={specified_gender}")
            
            # Get matching connectors
            connectors = self.find_by_series_and_pins(search_term, pin_count)
            
            # If gender was specified, filter by gender
            if specified_gender and connectors:
                connectors = [conn for conn in connectors if conn['gender'] == specified_gender]
            
            if connectors:
                results['connectors'] = connectors
                
                # Generate a more specific explanation based on what was found
                if specified_series and specified_gender:
                    explanation = f"Found {len(connectors)} {specified_series} {specified_gender} connectors with {pin_count} pins."
                elif specified_series:
                    explanation = f"Found {len(connectors)} {specified_series} connectors with {pin_count} pins."
                else:
                    explanation = f"Found {len(connectors)} {family} connectors with {pin_count} pins."
                
                # Generate comparison
                comparison_df = self.compare_connectors(connectors)
                results['comparison'] = comparison_df.to_dict('records')
            else:
                # Provide a more informative "not found" message
                if specified_series and specified_gender:
                    explanation = f"No {specified_series} {specified_gender} connectors found with {pin_count} pins."
                elif specified_series:
                    explanation = f"No {specified_series} connectors found with {pin_count} pins."
                else:
                    explanation = f"No {family} connectors found with {pin_count} pins."
                
                # Suggest fallback to general family
                if specified_series and not specified_gender:
                    general_connectors = self.find_by_series_and_pins(family, pin_count)
                    if general_connectors:
                        explanation += f" However, I found {len(general_connectors)} {family} connectors with {pin_count} pins."
                        results['connectors'] = general_connectors
                        comparison_df = self.compare_connectors(general_connectors)
                        results['comparison'] = comparison_df.to_dict('records')
        
        # Case 2: Pin count with dimension constraints
        elif pin_counts and (max_length is not None or max_height is not None):
            pin_count = list(pin_counts)[0]
            connectors = self.find_within_dimensions(
                max_length=max_length, 
                max_height=max_height,
                pin_count=pin_count
            )
            
            if connectors:
                results['connectors'] = connectors
                constraint_desc = []
                if max_length is not None:
                    constraint_desc.append(f"length ≤ {max_length}mm")
                if max_height is not None:
                    constraint_desc.append(f"height ≤ {max_height}mm")
                    
                constraint_str = " and ".join(constraint_desc)
                explanation = f"Found {len(connectors)} connectors with {pin_count} pins and {constraint_str}."
                
                # Generate comparison
                comparison_df = self.compare_connectors(connectors)
                results['comparison'] = comparison_df.to_dict('records')
            else:
                explanation = f"No connectors found with {pin_count} pins that meet the dimensional constraints."
        
        # Case 3: Optimal dimensions for pin count
        elif pin_counts and optimal_query:
            pin_count = list(pin_counts)[0]
            optimal_connector = self.find_optimal_by_dimension(pin_count, 'area')
            
            if optimal_connector:
                results['optimal_connector'] = optimal_connector
                area = optimal_connector['length'] * optimal_connector['height']
                explanation = (f"The connector with the smallest dimensions for {pin_count} pins is "
                            f"{optimal_connector['connector_family']}{optimal_connector['series']} "
                            f"{optimal_connector['gender']} ({optimal_connector['dimensions']}), "
                            f"with an area of {area:.1f} mm².")
            else:
                explanation = f"No connectors found with {pin_count} pins."
        
        # Case 4: General dimension query for a connector family
        elif families and dimension_query:
            family = families[0]
            # If a specific series was mentioned (e.g., "CMM220"), use that
            search_term = specified_series if specified_series else family
            all_family_connectors = self.get_all_connectors_for_family(search_term)
            
            # If gender was specified, filter by gender
            if specified_gender and all_family_connectors:
                all_family_connectors = [conn for conn in all_family_connectors if conn['gender'] == specified_gender]
            
            if all_family_connectors:
                # Group by series and gender
                series_data = {}
                for conn in all_family_connectors:
                    key = f"{conn['connector_family']}{conn['series']} {conn['gender']}"
                    if key not in series_data:
                        series_data[key] = []
                    series_data[key].append(conn)
                
                results['series_data'] = series_data
                
                if specified_series and specified_gender:
                    explanation = f"Found dimension data for {len(series_data)} {specified_series} {specified_gender} connector series."
                elif specified_series:
                    explanation = f"Found dimension data for {len(series_data)} {specified_series} connector series."
                else:
                    explanation = f"Found dimension data for {len(series_data)} {family} connector series."
            else:
                if specified_series and specified_gender:
                    explanation = f"No dimension data found for {specified_series} {specified_gender} connectors."
                elif specified_series:
                    explanation = f"No dimension data found for {specified_series} connectors."
                else:
                    explanation = f"No dimension data found for {family} connectors."
        
        # Default case: Not enough information
        else:
            if pin_counts:
                pin_count = list(pin_counts)[0]
                all_with_pins = self.find_by_pins(pin_count)
                
                if all_with_pins:
                    results['connectors'] = all_with_pins
                    explanation = f"Found {len(all_with_pins)} connectors with {pin_count} pins across all families."
                    
                    # Generate comparison
                    comparison_df = self.compare_connectors(all_with_pins)
                    results['comparison'] = comparison_df.to_dict('records')
                else:
                    explanation = f"No connectors found with {pin_count} pins."
            else:
                explanation = "Empty Response"
        
        return {'results': results, 'explanation': explanation}

    def generate_response(self, query: str) -> str:
        query_result = self.process_query(query)
        explanation = query_result['explanation']
        results = query_result.get('results', {})
        
        response_parts = [explanation]
        
        # Format connector comparison if available
        if 'comparison' in results:
            comparison = results['comparison']
            if comparison:
                response_parts.append("\n\nHere are the connector options:")
                
                # Format the table
                table_rows = []
                headers = list(comparison[0].keys())
                table_rows.append(" | ".join(headers))
                table_rows.append(" | ".join(["-" * len(header) for header in headers]))
                
                for row in comparison:
                    table_rows.append(" | ".join([str(row[header]) for header in headers]))
                
                response_parts.append("\n".join(table_rows))
        
        # Format optimal connector if available
        if 'optimal_connector' in query_result['results']:
            conn = query_result['results']['optimal_connector']
            response_parts.append(f"\nThe connector with the smallest dimensions is {conn['connector_family']}{conn['series']} {conn['gender']}:")
            response_parts.append(f"- Pin Count: {conn['pin_count']}")
            response_parts.append(f"- Dimensions: {conn['dimensions']}")
            response_parts.append(f"- Area: {round(conn['length'] * conn['height'], 2)} mm²")
        
        # NEW: Format max/min pin count results if available
        if 'max_pin_count' in results:
            pin_count = results['max_pin_count']
            connectors = results.get('max_pin_connectors', [])
            if connectors:
                conn = connectors[0]  # Use the first connector as example
                response_parts.append(f"\nDetails for the {conn['connector_family']}{conn['series']} {conn['gender']} with {pin_count} contacts:")
                response_parts.append(f"- Dimensions: {conn['dimensions']}")
                response_parts.append(f"- Area: {round(conn['length'] * conn['height'], 2)} mm²")
        
        if 'min_pin_count' in results:
            pin_count = results['min_pin_count']
            connectors = results.get('min_pin_connectors', [])
            if connectors:
                conn = connectors[0]  # Use the first connector as example
                response_parts.append(f"\nDetails for the {conn['connector_family']}{conn['series']} {conn['gender']} with {pin_count} contacts:")
                response_parts.append(f"- Dimensions: {conn['dimensions']}")
                response_parts.append(f"- Area: {round(conn['length'] * conn['height'], 2)} mm²")
        
        # Format series data if available
        if 'series_data' in results:
            series_data = results['series_data']
            for series_name, connectors in series_data.items():
                # Get pin count range
                pin_counts = sorted(set(conn['pin_count'] for conn in connectors))
                min_pin = min(pin_counts)
                max_pin = max(pin_counts)
                
                # Get dimension ranges
                min_length = min(conn['length'] for conn in connectors)
                max_length = max(conn['length'] for conn in connectors)
                min_height = min(conn['height'] for conn in connectors)
                max_height = max(conn['height'] for conn in connectors)
                
                response_parts.append(f"\n{series_name}:")
                response_parts.append(f"- Pin count range: {min_pin} to {max_pin}")
                response_parts.append(f"- Length range: {min_length} to {max_length} mm")
                response_parts.append(f"- Height range: {min_height} to {max_height} mm")
        
        return "\n".join(response_parts)
    def generate_response(self, query: str) -> str:
        query_result = self.process_query(query)
        explanation = query_result['explanation']
        results = query_result.get('results', {})
        
        response_parts = [explanation]
        
        # Format connector comparison if available
        if 'comparison' in results:
            comparison = results['comparison']
            if comparison:
                response_parts.append("\n\nHere are the connector options:")
                
                # Format the table
                table_rows = []
                headers = list(comparison[0].keys())
                table_rows.append(" | ".join(headers))
                table_rows.append(" | ".join(["-" * len(header) for header in headers]))
                
                for row in comparison:
                    table_rows.append(" | ".join([str(row[header]) for header in headers]))
                
                response_parts.append("\n".join(table_rows))
        
        # Format optimal connector if available
        if 'optimal_connector' in query_result['results']:
            conn = query_result['results']['optimal_connector']
            response_parts.append(f"\nThe connector with the smallest dimensions is {conn['connector_family']}{conn['series']} {conn['gender']}:")
            response_parts.append(f"- Pin Count: {conn['pin_count']}")
            response_parts.append(f"- Dimensions: {conn['dimensions']}")
            response_parts.append(f"- Area: {round(conn['length'] * conn['height'], 2)} mm²")
        
        # Format series data if available
        if 'series_data' in results:
            series_data = results['series_data']
            for series_name, connectors in series_data.items():
                # Get pin count range
                pin_counts = sorted(set(conn['pin_count'] for conn in connectors))
                min_pin = min(pin_counts)
                max_pin = max(pin_counts)
                
                # Get dimension ranges
                min_length = min(conn['length'] for conn in connectors)
                max_length = max(conn['length'] for conn in connectors)
                min_height = min(conn['height'] for conn in connectors)
                max_height = max(conn['height'] for conn in connectors)
                
                response_parts.append(f"\n{series_name}:")
                response_parts.append(f"- Pin count range: {min_pin} to {max_pin}")
                response_parts.append(f"- Length range: {min_length} to {max_length} mm")
                response_parts.append(f"- Height range: {min_height} to {max_height} mm")
        
        return "\n".join(response_parts)