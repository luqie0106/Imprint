# ACR 17.2 ARM64 phase3 selected reconstruction evidence

Read-only ghidra-project; program `/Camera_Raw_arm64.analysis-383152`. No Ghidra edits.
Offsets below are character ranges in this decompilation, not instruction addresses.

## FUN_013c8bc8 parameter-derived bias and clipping

```c
iVar7 = 1 << (ulong)(((uint)param_16 - 1) + param_16._4_4_ & 0x1f);
    iVar8 = 1 << (ulong)((uint)param_16 - 1 & 0x1f);
    if ((int)(uint)param_16 < 1) {
      iVar8 = 0;
    }
    iVar10 = -(uint)param_16;
    uVar11 = (2 << (ulong)(param_16._4_4_ - 1U & 0x1f)) - 1;
    if (param_16._4_4_ == 0) {
      uVar11 = 0xffffffff;
    }
    iVar1 = (param_10._4_4_ - 5U & 0xfffffffc) + 4;
    lVar37 = (long)(int)(uint)param_10;
    uVar38 = -(ulong)((uint)param_10 >> 0x1f
```

## FUN_013c8bc8 scalar reconstruction (bounded excerpt)

```c
< param_10._4_4_) {
LAB_013c9214:
          lVar51 = 0;
          iVar54 = puVar47[-(ulong)(iVar54 != 0)];
          iVar75 = *(int *)*pauVar48;
          do {
            piVar6 = (int *)(*pauVar48 + lVar51);
            iVar76 = *(int *)((long)puVar47 + lVar51);
            iVar57 = *(int *)((long)puVar41 + lVar51);
            piVar5 = piVar6;
            if (iVar50 != 1) {
              piVar5 = piVar6 + 1;
            }
            iVar62 = *piVar5;
            iVar63 = *(int *)(lVar44 + lVar51);
            if (puVar46 != (undefined8 *)0x0) {
              if (param_12._4_4_ == 0 || uVar34 != 0) {
                piVar6 = (int *)(lVar43 + lVar51);
              }
              uVar31 = *(int *)(lVar49 + lVar51) + iVar7 + (iVar54 + iVar75 + iVar76 + *piVar6 >> 2)
              ;
              if (*param_15 != '\0') {
                uVar4 = uVar31 & ((int)uVar31 >> 0x1f ^ 0xffffffffU);
                if (*param_14 <= (int)uVar31) {
                  uVar4 = *param_14 - 1;
                }
                uVar31 = *(uint *)(*(long *)(param_14 + 2) + (long)(int)uVar4 * 4);
              }
              uVar31 = (int)(uVar31 + iVar8) >> ((uint)param_16 & 0x1f);
              uVar4 = uVar11;
              if ((int)uVar31 <= (int)uVar11) {
                uVar4 = uVar31 & ((int)uVar31 >> 0x1f ^ 0xffffffffU);
              }
              *(short *)puVar46 = (short)uVar4;
              uVar31 = iVar76 + iVar7;
              if (*param_15 != '\0') {
                uVar4 = uVar31 & ((int)uVar31 >> 0x1f ^ 0xffffffffU);
                if (*param_14 <= (int)uVar31) {
                  uVar4 = *param_14 - 1;
                }
                uVar31 = *(uint *)(*(long *)(param_14 + 2) + (long)(int)uVar4 * 4);
              }
              uVar31 = (int)(uVar31 + iVar8) >> ((uint)param_16 & 0x1f);
              uVar4 = uVar11;
              if ((int)uVar31 <= (int)uVar11) {
                uVar4 = uVar31 & ((int)uVar31 >> 0x1f ^ 0xffffffffU);
              }
              *(short *)((long)puVar46 + 2) = (short)uVar4;
              puVar46 = (undefined8 *)((long)puVar46 + 4);
            }
            if (puVar29 != (undefined8 *)0x0) {
              uVar31 = iVar75 + iVar7;
              if (*param_15 != '\0') {
                uVar4 = uVar31 & ((int)uVar31 >> 0x1f ^ 0xffffffffU);
                if (*param_14 <= (int)uVar31) {
                  uVar4 = *param_14 - 1;
                }
                uVar31 = *(uint *)(*(long *)(param_14 + 2) + (long)(int)uVar4 * 4);
              }
              uVar31 = (int)(uVar31 + iVar8) >> ((uint)param_16 & 0x1f);
              uVar4 = uVar11;
              if ((int)uVar31 <= (int)uVar11) {
                uVar4 = uVar31 & ((int)uVar31 >> 0x1f ^ 0xffffffffU);
              }
              *(short *)puVar29 = (short)uVar4;
              uVar31 = iVar63 + iVar7 + (
```

## Selected ARM64 instructions checked independently

```asm
013c92a8 add w6,w6,w3
013c92ac add w6,w6,w1
013c92b0 add w6,w6,w7
013c92b8 add w30,w30,w28
013c92bc add w6,w30,w6, ASR #0x2
013c92dc ldr w6,[x7, w6, SXTW #0x2]
013c92e0 add w6,w6,w25
013c92e4 asr w6,w6,w22
013c92f4 strh w6,[x10]
013c9334 strh w6,[x10, #0x2]
013c9338 add x10,x10,#0x4
013c9368 add w3,w1,w3
013c936c add w3,w3,w5
013c9370 add w3,w3,w2
013c9374 asr w3,w3,#0x2
013c938c strh w5,[x8]
```

## Callsite clarification

`013e1bc8` and `013e2414` are instruction addresses inside FUN_013e0b84, not separate helpers. Both have `bl 0x013e3d84`. Decompiling either interior address returns the containing 77,519-character function; only bounded fragments were inspected. The phase2 report already identifies them as callsites.

## Header WGT length and array initialization

```c
                   bVar3 = *(byte *)((long)param_2 + uVar23 + 3);
                          uVar20 = *(uint *)(lVar27 + 0x94);
                          if (uVar20 * 2 + 2 !=
                              ((uint)bVar3 |
                              (CONCAT11(bVar3,*(undefined1 *)((long)param_2 + uVar23 + 2)) &
                              0xff00ff) << 8)) {
                            uVar18 = 0xffff5eeb;
                            goto LAB_013d334c;
                          }
                          *(undefined8 *)(lVar27 + 0xd8) = 0;
                          *(undefined8 *)(lVar27 + 0xd0) = 0;
                          *(undefined8 *)(lVar27 + 200) = 0;
                          *(undefined8 *)(lVar27 + 0xc0) = 0;
                          *(undefined8 *)(lVar27 + 0xb8) = 0;
                          *(undefined8 *)(lVar27 + 0xb0) = 0;
                          *(undefined8 *)(lVar27 + 0xa8) = 0;
                          *(undefined8 *)(lVar27 + 0xa0) = 0;
                          *(undefined8 *)(lVar27 + 0x98) = 0;
                          *(undefined8 *)(lVar27 + 0xe8) = 0x4848484848484848;

```

