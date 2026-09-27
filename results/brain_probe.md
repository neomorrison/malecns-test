# maleCNS spiking-model probe

Brain model: 141,615 neurons. Each channel driven at 100 Hz Poisson for 1000 ms, mean of 3 seeds.
Rates in Hz: mean over the left / right cells of each readout group.

```
sugar GRNs          77        (LB3a, LB3b, LB3c, LB3d)
pursuit VPNs L/R   369 / 360  (LC10a, LC10c-1, LC10d, LC10e)
loom VPNs    L/R   165 / 146  (LPLC2, LC4)
food ORNs    L/R   108 / 124  (ORN_DM1, ORN_DM2, ORN_DM4, ORN_VA2)
readout GF        L/R 1/1  (DNp01)
readout STEER     L/R 3/3  (DNa02, DNa01, DNg13)
readout FORWARD   L/R 3/3  (DNp09, DNg97, DNg100)
readout BACKWARD  L/R 2/2  (MDN)
readout FEED      L/R 1/1  (MN9)
```

| stimulus | GF | STEER | FORWARD | BACKWARD | FEED |
|---|---|---|---|---|---|
| none | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| sugar | 0 / 0 | 0 / 0 | 1 / 1 | 0 / 0 | 162 / 7 |
| target_left | 0 / 0 | 47 / 34 | 0 / 1 | 0 / 0 | 0 / 0 |
| target_right | 0 / 0 | 25 / 73 | 1 / 1 | 0 / 0 | 52 / 0 |
| loom_left | 398 / 293 | 0 / 31 | 0 / 0 | 0 / 7 | 0 / 0 |
| loom_right | 336 / 397 | 0 / 0 | 0 / 0 | 4 / 0 | 0 / 0 |
| hunger_drive | 0 / 1 | 22 / 11 | 78 / 83 | 47 / 41 | 0 / 0 |
| odor_left | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| odor_right | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

## Most active descending neurons per stimulus

- **none**: 
- **sugar**: DNge059_L (152), DNge031_L (149), DNge031_R (148), DNg67_R (136), DNge059_R (123), DNg67_L (121), DNge106_R (108), DNge007_L (101), DNge007_R (100), DNg49_L (88)
- **target_left**: DNg33_R (290), DNg33_L (289), pIP1_L (190), DNp31_R (143), DNp31_L (138), DNg111(hb1573072464)_L (123), DNg74_a_R (123), DNg105_L (122), DNg74_a_L (121), DNg105_R (120)
- **target_right**: DNg33_R (285), DNg33_L (285), pIP1_R (231), DNge031_R (223), DNg105_R (215), DNg105_L (191), DNp31_R (180), DNge031_L (151), DNg111(hb1573072464)_R (149), DNge101_R (139)
- **loom_left**: DNp01(GF)_L (398), DNp04_L (398), DNp103(PVLP119)_L (361), DNp02_L (360), DNp11_L (347), DNg40(hb5813056435)_L (335), DNp06_L (312), DNp03_L (304), DNp01(GF)_R (293), DNge119_R (288)
- **loom_right**: DNp01(GF)_R (397), DNp04_R (396), DNp103(PVLP119)_R (360), DNp01(GF)_L (336), DNp02_R (336), DNg40(hb5813056435)_R (335), DNp11_R (326), DNg74_a_L (323), DNg74_a_R (319), DNg108_R (296)
- **hunger_drive**: DNg33_R (184), DNg33_L (184), DNa06(PS039)_L (140), DNge006_L (120), DNg75(hb1874217622)_L (118), DNge033_L (103), DNg49(hb1405978277)_R (100), DNp09_R (99), DNg100_R (98), DNg49_L (98)
- **odor_left**: DNb05_R (90), DNb05_L (79), DNg56_R (17), DNg35_L (10), DNg35_R (8)
- **odor_right**: DNb05_R (118), DNb05_L (83), DNg56_R (33), DNg35_R (26), DNg35_L (19), DNg99_R (17)
