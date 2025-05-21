<!-- PAGE: 1 -->
# NICOMATIC Test report summary
# CMM Family
# DERATING CURVES
# (Current carrying capacity)
<!-- PAGE: 2 -->

## I. Introduction

### A. Purpose
The CMM connectors' family are manufactured to meet or exceed the requirements of MIL-DTL-55302G standard.

### B. Scope
Define the max current allowable without any temperature elevation greater than 30°C for LF contacts and 40°C for HP contacts, depending on the number of contacts of a CMM connector.

The following data has been taken from NICOMATIC Qualification test reports QTR1018 and QTR1103.

### C. Conclusion
The CMM connectors' family are qualified regarding DERATING CURVES (Current carrying capacity) according to IEC 60512-5-2 Test 5b.

| Max Current allowable in continue @ 25°C (A) | For 1 Contact | For max contact on connector |
|---------------------------------------------|--------------|---------------------------|
| LF contacts                                 | 12 A         | 3 A                       |
| HP22                                        | 23 A         | 10 A                      |
| HP30                                        | 34 A         | 20 A                      |

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
<!-- PAGE: 3 -->
- 222Y08SXX-0004-4300CMM – HP30 + LF female contacts Straight PCB _ 30-4300-CMM + C14764
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
According to IEC 60512-5-2 Test 5b, DERATING CURVES (Current carrying capacity):

The temperature limits allowable during the current carrying:
- LF: 30°C
- HP: 40°C
<!-- PAGE: 4 -->
### C. Test Method and Results
The specimen shall be arranged in the enclosure as described in 3.1 and its terminals are connected to a regulated power supply through an ammeter, see figure 3.

The loading current may be a.c. or d.c. When a.c. current is used, the r.m.s. value applies.

If d.c. current is used, avoid voltage bias influence on the thermocouple by executing the test with reverse current.

The current shall be maintained for a period of approximately 1 h after thermal stability is achieved at each of the selected current levels. This is defined as when three consecutive values of temperature rise, taken at 5 min intervals, do not differ by more than 2 K of each other.

#### Max Current in continue vs number of LF contacts

| Number of contacts | 1   | 9  | 15 | 21  | 30  | 36  | 39  | 51 | 69 | ... |
|-------------------|-----|----|----|-----|-----|-----|-----|----|----|----|
| Current (A)       | 12  | 6  | 5  | 4.5 | 3.5 | 3.5 | 3.5 | 3  | 3  | 3  |

*Average temperature elevation 30°C @ Max Current*
<!-- PAGE: 5 -->
#### Max Current in continue vs number of HP22 contacts

| Number of contacts | 2   | 4   | 6   | 8   | 10  | 12  | 14  | 16  | 18  | 20  | 22  | 24  | 26  | ... |
|-------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Current (A)       | 23  | 20  | 19  | 17  | 16  | 15  | 13  | 12  | 11  | 10  | 10  | 10  | 10  | 10  |

*Average temperature elevation 40°C @ Max Current*

#### Max Current in continue vs number of HP30 contacts

| Number of contacts | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10  | ... |
|-------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Current (A)       | 34  | 32  | 30  | 28  | 26  | 24  | 22  | 21  | 20  | 20  | 20  |

*Average temperature elevation 40°C @ Max Current*

---

*Date: 22.11.17*