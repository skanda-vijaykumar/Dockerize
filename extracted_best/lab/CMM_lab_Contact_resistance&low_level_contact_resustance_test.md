
<!-- PAGE: 1 -->
# NICOMATIC Test report summary

## CMM Family

# CONTACT RESISTANCE AND LOW LEVEL CONTACT RESISTANCE Test
<!-- PAGE: 2 -->
*21.12.16*

## I. Introduction

### A. Purpose

The CMM connectors' family are manufactured to meet or exceed the requirements of MIL-DTL-55302G standard.

### B. Scope

Determine the resistance of mated connector contacts attached to lengths of wire by measuring the voltage drop across the contacts while they are carrying a specified current.

The following data has been taken from NICOMATIC Qualification test reports QTR0804 and QTR0805.

### C. Conclusion

The CMM connectors' family are qualified regarding CONTACT RESISTANCE and LOW LEVEL CONTACT RESISTANCE according to MIL-DTL-55302G.

| | Contact Resistance | Low Level Contact Resistance |
|---|---|---|
| LF Contacts | < 8.7 mOhm | < 10 mOhm |
| HP22 Contacts | < 3 mOhm max | < 3 mOhm |
| HP30 Contacts | < 3 mOhm max | < 3 mOhm |

## II. Test Method and Requirements

### A. List of Test Samples

#### a. CMM 200 Series
- 201Y50L – LF male contacts Straight PCB _ 13507
- 202Y50 – LF female contacts Straight PCB _ C14764

#### b. CMM 220 Series
- 221V50FXX – LF male contacts 90° PCB _ 13507
- 222S50MXX – LF female crimp contacts _ C12468
- 222YL26MXX – LF male contacts Straight PCB _ C14810
- 221S26FXX – LF male crimp contacts _ 12969
- 221D00FXX-0008-3400CMM – HP30 male contacts 90° PCB _ 30-3400-CMM
- 222E00MXX-0008-4320 – HP30 female straight contacts on cable _ 30-4320
- 222Y08SXX-0004-4300CMM – HP30 + LF female contacts Straight PCB _ 30-4300-CMM + C14764
<!-- PAGE: 3 -->
- 221S08FXX-0004-3308 – HP30 + LF male contacts Straight on cable _ 30-3308 + 12969
- 221S06FXX-0003-3320 – HP30 + LF male contacts Straight on cable _ 30-3320 + 12969
- 222S06MXX-0003-4308 – HP30 + LF female contacts Straight on cable _ 30-4308 + C12468

#### c. CMM 320 Series
- 321C057FXX – LF male crimp contacts _ 12960
- 322C057MXX – LF female crimp contact _ C13064-P
- 321V096FXX – LF male contacts 90° PCB _ 13507
- 322Y096MXX – LF female contacts Straight PCB _ C14812
- 341D000FXX-0018-340014 – HP22 male contacts 90° PCB _ 22-3400-XX
- 342E000MXX-0018-4310 – HP22 female straight contacts on cable _ 22-4310
- 342D000MXX-0048-430014 – HP22 female contacts Straight PCB _ 22-4300-14
- 341E000FXX-0048-3310 – HP22 male straight contacts on cable _ 22-3310

### B. Requirements

According to MIL-DTL-55302G standard and EIA-364-06C / EIA-364-23C test procedures:

Rem.: There no specifications regarding the higher currents for power contacts

#### Contact Resistance:

| Wire size, AWG type E as specified in MIL-DTL-16878 | Test current (AMP) | Maximum contact resistance (mΩ) | Maximum potential drop (mV) |
|---|---|---|---|
| 20 | 7.5 | 6.0 | 45.0 |
| 22 | 5.0 | 8.0 | 40.0 |
| 24 | 3.0 | 8.7 | 26.0 |
| 26 | 2.0 | 12.0 | 24.0 |
| 28 | 1.5 | 14.7 | 22.0 |
| 30 | 1.0 | 20.0 | 20.0 |

#### Low Level Contact Resistance:

| Wire size, AWG type E per MIL-DTL-16878 | Maximum test current (AMP) | Maximum contact resistance (mΩ) | Maximum potential drop (mV) |
|---|---|---|---|
| 20 | 0.1 | 9 | 0.9 |
| 22 | " | 15 | 1.5 |
| 24 | " | 20 | 2.0 |
| 26 | " | 25 | 2.5 |
| 28 | " | 40 | 4.0 |
| 30 | " | 50 | 5.0 |
<!-- PAGE: 4 -->
### C. Test Method and Results

#### a. Contact Resistance test method

Energize the circuit and increase the current until the required test current is achieved. The lowest voltage shall be used that allows the specified test current to be achieved.

Connect the voltmeter probes (leads) to the specimen (if not permanently attached) and measure and record the voltage drop. Assure that the test current has remained at the correct value.

When readings are one millivolt or less on small dc measurements, reverse current readings shall be taken. The two measurements shall be averaged to cancel the effects of thermal potentials.

Measure and record the reverse voltage drop.

Calculate the specimen voltage drop as follows:

Specimen voltage drop = (forward voltage drop + reverse voltage drop) / 2

Calculate resistance, if required.

#### b. Low Level Contact Resistance test method

The following option will be used to correct for thermal EMF's: 4 wire micro-ohmmeter.

The micro-ohmmeter employs a method to correct for thermal EMF.

The following details shall apply:

a. Method of connection - Attach current-voltage leads at extreme ends of contacts. For crimp type contacts, attach current-voltage leads to wires, at closest point to contact without touching contact.

b. 100 milliamperes dc maximum.

Measure and record the contact resistance of the specimen under test with a test current of 100 milliamperes maximum and 20 millivolts open circuit (source) voltage maximum.

## References

| References | Max Contact Resistance (mOhm) | Max Low Level Contact Resistance (mOhm) |
|---|---|---|
| **LF Contacts** | | |
| 201Y50L / 202V50FXX | 5.73 | 5.3 |
| 221V50FXX / 222S50MXX | 5.38 | 5.26 |
| 221S26FXX / 222YL26MXX | 4.98 | 4.05 |
| 321C057FXX / 322C057MXX | 3.28 | 3.45 |
| 321V096FXX / 322Y096MXX | 5.88 | 8.7 |
| **Contacts HP 30 series** | | |
| 221D00FXX-0008-3400CMM / 222E00MXX-0008-4320 | 0.67 | 0.84 |
| 221S08FXX-0004-3308 / 222Y08SXX-0004-4300CMM | LF: 4.48<br>HP: 0.91 | LF: 5.19<br>HP: 0.86 |
| 221S06FXX-0003-3320 / 222S06MXX-0003-4308 | LF: 2.30<br>HP: 0.17 | LF: 4.18<br>HP: 0.88 |
| **Contacts HP 22 series** | | |
| 341D000FXX-0018-340014 / 342E00MXX-0018-4310 | 1.42 | 1.57 |
| 341E000FXX-0048-3310 / 342D000MXX-0048-430014 | 1.15 | 1.08 |

Here are the wires size used for this test:
- LF contacts 3A: AWG 24
- HP contacts 8A/10A: AWG 16
- HP contacts 20A: AWG 12