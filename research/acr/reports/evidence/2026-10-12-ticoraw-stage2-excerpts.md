# TicoRAW stage 2 — bounded Ghidra-project excerpts

2026-10-12. Decompiled pseudocode is not original source and signatures can be incomplete. Only selected portions of large functions are retained. Slices use response character positions, not VM addresses.

## scheduler tasks — FUN_01502bb8

```c
CONCAT44(uStack_ec,uStack_f0);
    *(ulong *)(param_1 + 0x58) = CONCAT44(uStack_f4,local_f8);
    *(undefined8 *)(param_1 + 0x70) = uStack_e0;
    *(ulong *)(param_1 + 0x68) = local_e8;
    *(undefined8 *)(param_1 + 0x80) = uStack_d0;
    *(undefined8 *)(param_1 + 0x78) = uStack_d8;
    FUN_014d61cc(*(undefined8 *)(param_1 + 0x28));
    uVar7 = FUN_015026a8(param_1,puVar1,&local_80,auStack_90);
    if (-1 < (int)uVar7) {
      FUN_014d5354(*(undefined8 *)(param_1 + 0x30),uVar7,0);
      puVar1 = &DAT_081985a0;
      if (uVar20 != 0) {
        puVar1 = &DAT_083afb98;
      }
      uVar11 = (ulong)local_7c;
      if ((int)local_7c < 1) {
        iVar5 = 0;
        if (local_78 < 1) {
          uVar11 = 0;
          uVar17 = 0;
        }
        else {
          uVar17 = 0;
          lVar14 = 0xc;
          do {
            **(undefined8 **)(*(long *)(param_1 + 0x30) + uVar17 * 8) = FUN_0157e768;
            *(long *)(*(long *)(*(long *)(param_1 + 0x30) + uVar17 * 8) + 8) =
                 *(long *)(param_1 + 0x38) + lVar14 + -0xc;
            *(undefined8 *)(*(long *)(param_1 + 0x38) + lVar14 + -0xc) = uVar19;
            *(int *)(*(long *)(param_1 + 0x38) + lVar14 + -4) = iVar5;
            *(int *)(*(long *)(param_1 + 0x38) + lVar14) = local_88;
            if ((int)uVar17 - local_78 == -1) {
              *(int *)(*(long *)(param_1 + 0x38) + lVar14) =
                   *(int *)(lVar18 + 0xc) - local_88 * (local_78 + -1);
            }
            uVar17 = uVar17 + 1;
            lVar14 = lVar14 + 0x30;
            iVar5 = iVar5 + local_88;
          } while ((int)uVar17 != local_78);
          uVar11 = 0;
          iVar5 = 1;
        }
      }
      else {
        iVar12 = 0;
        iVar5 = 0;
        lVar13 = 0;
        lVar14 = 0;
        uVar15 = *puVar1;
```

## scheduler submission — FUN_01502bb8

```c
          iVar3 = *(int *)(lVar10 + 0x18);
                *(int *)(lVar10 + 0x18) = iVar3 + 1;
                *(int *)(*(long *)(lVar10 + 0x20) + (long)iVar3 * 4) = (int)uVar9;
                *(int *)(*(long *)(lVar21 + lVar13) + 0x14) =
                     *(int *)(*(long *)(lVar21 + lVar13) + 0x14) + 1;
              }
              uVar20 = (int)uVar9 + 1;
              lVar13 = lVar13 + 8;
              uVar9 = (ulong)uVar20;
              lVar8 = lVar8 + 0x30;
            } while ((int)uVar20 < local_78 + (int)uVar11);
          }
          iVar16 = local_84 + iVar16;
          iVar12 = iVar12 + 1;
          uVar17 = uVar17 + 1;
        } while (iVar12 != local_74);
        uVar7 = uVar7 & 0xffffffff;
      }
      if (local_80 != 1) {
        FUN_01e79a98();
        goto LAB_015033a0;
      }
      lVar18 = (uVar17 & 0xffffffff) * 8;
      **(undefined8 **)(*(long *)(param_1 + 0x30) + lVar18) = FUN_013ceb54;
      lVar14 = ((uVar17 & 0xffffffff) + (uVar17 & 0xffffffff) * 2) * 0x10;
      *(long *)(*(long *)(*(long *)(param_1 + 0x30) + lVar18) + 8) =
           *(long *)(param_1 + 0x38) + lVar14;
      *(undefined8 *)(*(long *)(param_1 + 0x38) + lVar14) = uVar19;
      *(uint *)(*(long *)(param_1 + 0x38) + lVar14 + 8) = local_7c;
      uVar7 = FUN_014d5df4(*(undefined8 *)(param_1 + 0x28),*(undefined8 *)(param_1 + 0x30),uVar7);
      if ((int)uVar7 == 0) goto LAB_01503338;
    }
  }
  if (*(long *)PTR____stack_chk_guard_07a703a0 == local_70) {
    return uVar7;
  }
LAB_015033a0:
                    /* WARNING: Subroutine does not return */
  ___stack_chk_fail();
}
```

## pool job copy — FUN_014d5df4

```c
l_50);
  }
  uVar2 = FUN_014d5380(param_1,param_3);
  if ((int)uVar2 == 0) {
    *(int *)(param_1 + 0x30) = (int)param_3;
    if ((int)param_3 != 0) {
      uVar4 = 0;
      do {
        puVar5 = *(undefined8 **)(*(long *)(param_1 + 0x28) + uVar4 * 8);
        puVar3 = *(undefined8 **)(param_2 + uVar4 * 8);
        uVar1 = *(undefined4 *)(param_1 + 0x1c);
        *puVar5 = *puVar3;
        puVar5[1] = puVar3[1];
        *(undefined4 *)(puVar5 + 2) = *(undefined4 *)(puVar3 + 2);
        *(undefined4 *)((long)puVar5 + 0x14) = *(undefined4 *)((long)puVar3 + 0x14);
        *(undefined4 *)(puVar5 + 3) = *(undefined4 *)(puVar3 + 3);
        _memcpy((void *)puVar5[4],(void *)puVar3[4],(long)*(int *)(puVar3 + 3) << 2);
        *(undefined1 *)(puVar5 + 5) = 0;
        *(undefined4 *)((long)puVar5 + 0x2c) = uVar1;
        uVar4 = uVar4 + 1;
      } while (uVar4 < *(uint *)(param_1 + 0x30));
    }
    _bzero(*(void **)(param_1 + 0x38),(ulong)*(uint *)(param_1 + 4) << 2);
    _bzero(*(void **)(param_1 + 0x40),(ulong)*(uint *)(param_1 + 4) << 2);
    *(undefined4 *)(param_1 + 0x50) = 0;
    _bzer
```

## pool worker binding — FUN_014d54f0

```c
ar1 + 10);
            if (*(long *)((long)pvVar2 + uVar8 * 8) == 0) goto LAB_014d5700;
            uVar8 = uVar8 + 1;
            param_3 = (ulong)(uint)puVar1[3];
          } while (uVar8 < param_3);
          param_1 = puVar1[2];
        }
        puVar1[6] = 0;
        pvVar2 = _malloc((ulong)param_1 << 3);
        *(void **)(puVar1 + 4) = pvVar2;
        if (pvVar2 != (void *)0x0) {
          if (param_1 == 0) {
            return puVar1;
          }
          uVar8 = 0;
          while( true ) {
            uVar6 = __Znwm(8);
            local_48 = puVar1;
            FUN_014d636c(uVar6,FUN_014d58b0,&local_48);
            *(undefined8 *)(*(long *)(puVar1 + 4) + uVar8 * 8) = uVar6;
            if (*(long *)(*(long *)(puVar1 + 4) + uVar8 * 8) == 0) break;
            uVar8 = uVar8 + 1;
            if ((uint)puVar1[2] <= uVar8) {
              return puVar1;
            }
          }
        }
      }
    }
LAB_014d5700:
    FUN_014d5740(puVar1);
  }
  return (undefined4 *)0x0;
}
```

## worker invocation — FUN_014d58b0

```c
:
    uVar7 = 1;
LAB_014d5dc8:
                    /* WARNING: Subroutine does not return */
    __ZNSt3__120__throw_system_errorEiPKc(uVar11,uVar7,pcVar8);
  }
  __ZNSt3__15mutex6unlockEv(local_70);
  local_68 = 0;
  uVar4 = _setjmp(piVar5);
  uVar6 = (ulong)uVar4;
  if (uVar4 == 0) {
    puVar9 = *(undefined8 **)(*(long *)(param_1 + 0x28) + uVar16 * 8);
    uVar6 = (*(code *)*puVar9)(iVar2,uVar19,puVar9[1]);
  }
  if (local_70 == 0) {
LAB_014d5da0:
    uVar7 = 1;
    pcVar8 = "unique_lock::lock: references null mutex";
    uVar11 = uVar6;
    goto LAB_014d5dc8;
  }
  uVar11 = (ulong)local_68;
  if (local_68 != 0) {
    uVar7 = 0xb;
    pcVar8 = "unique_lock::lock: already locked";
    goto LAB_014d5dc8;
  }
  __ZNSt3__15mutex4lockEv();
  *(undefined1 *)(*(long *)(*(long *)(param_1 + 0x28) + uVar16 * 8) + 0x28) = 0;
  *(int *)(*(long *)(param_1 + 0x40) + lVar17) = *(int *)(*(long *)(param_1 + 0x40) + lVar17) + 1;
  iVar14 = *(int *)(*(long *)(param_1 + 0x48) + lVar17);
  if ((int)uVar6 == 0) {
    if (iVar14 != 0) goto LAB_014d5b80;
    if (*(int *)(param_1 + 0x50) == 0) {
      lVar12 = *(long *)(param_1 + 0x28);
      lVar17 = *(long *)(lVar12 + uVar16 * 8);
      if (*(int *)(lVar17 + 0x18) < 1) {
        iVar14 = 0;
      }
      else {
        lVar13 = *(long *)(lVar17 + 0x20);
        lVar15 = 0;
        iVar14 = 0;
        do {
          lVar17 = *(long *)(lVar12 + (long)*(int *)(lVar13 + lVar15 * 4) * 8);
          *(int *)(lVar17 + 0x10) = *(int *)(lVar17 + 0x10) + -1;
          lVar12 = *(long *)(param_1 + 0x28);
          lVar17 = *(long *)(lVar12 + uVar16 * 8);
```

## ff20 scanner — FUN_013ceb54

```c


  lVar12 = *param_3;
  uVar1 = *(uint *)(param_3 + 1);
  lVar17 = *(long *)(lVar12 + 0x28);
  lVar10 = *(long *)(lVar17 + 0xc0);
  iVar7 = (int)param_2;
  lVar16 = *(long *)(lVar10 + (long)iVar7 * 0x48 + 0x28);
  if (0 < (int)uVar1) {
    uVar13 = 0;
    lVar18 = *(long *)(lVar10 + (long)iVar7 * 0x48 + 8);
    do {
      lVar9 = *(long *)(lVar17 + 0x38) + uVar13 * 0x30;
      iVar14 = *(int *)(lVar9 + 0x28);
      if (iVar14 < *(int *)(lVar9 + 0x2c) + iVar14 + *(int *)(lVar18 + 0x24) * 2) {
        lVar15 = 0;
        lVar9 = 0;
        do {
          if (*(int *)(lVar18 + 0xc) <= iVar14) break;
          iVar2 = *(int *)(lVar18 + 0x90);
          iVar3 = 0;
          if (iVar2 != 0) {
            iVar3 = iVar14 / iVar2;
          }
          if (iVar14 == iVar3 * iVar2) {
            puVar6 = (ushort *)FUN_013b6598(lVar12,param_2,lVar9 + lVar16,6);
            if (puVar6 == (ushort *)0x0) {
              return 0xffff5fe0;
            }
            if ((ushort)(*puVar6 >> 8 | *puVar6 << 8) != 0xff20) {
              return 0xffff5fdf;
            }
            if ((ushort)(puVar6[1] >> 8 | puVar6[1] << 8) != 4) {
              return 0xffff5fdf;
            }
            uVar4 = 0;
            if (*(int *)(lVar18 + 0x90) != 0) {
              uVar4 = iVar14 / *(int *)(lVar18 + 0x90);
            }
            if (uVar4 != ((uint)(puVar6[2] >> 8) | (puVar6[2] & 0xff00ff) << 8)) {
              return 0xffff5fdf;
            }
            pbVar5 = (byte *)FUN_013b6598(lVar12,param_2,lVar9 + 6 + lVar16,3);
            lVar11 = lVar9 + 6;
            lVar15 = lVar9;
          }
          else {
            pbVar5 = (byte *)FUN_013b6598(lVar12,param_2,lVar9 + lVar16,3);
            lVar11 = lVar9;
          }
          if (pbVar5 == (byte *)0x0) {
            return 0xffff5fde;
          }
          lVar9 = lVar11 + (ulong)(((uint)*pbVar5 << 0x10 | (uint)pbVar5[1] << 8 | (uint)pbVar5[2])
                                  + (*(int *)(lVar18 + 0x94) * 2 + 0x2fU >> 3));
          iVar14 = *(int *)(lVar18 + 0x24) + iVar14;
          lVar11 = *(long *)(lVar17 + 0x38) + uVar13 * 0x30;
        } while (iVar14 < *(int *)(lVar11 + 0x28) + *(int *)(lVar18 + 0x24) * 2 +
                          *(int *)(lVar11 + 0x2c));
      }
      else {
        lVar9 = 0;
        lVar15 = 0;
      }
      lVar11 = FUN_013b6598(lVar12,param_2,lVar16 + 6,lVar9 + -6);
      if (lVar1
```

## primary thunk — FUN_013e2a18

```c

void FUN_013e2a18(void)

{
  DAT_081985a0 = FUN_013e0b84;
  FUN_013e0b84();
  return;
}
```

## decode packet prefix — FUN_013e0b84

```c
0;
    puStack_128 = (ulong *)((long)puVar48 + 0xc);
    iVar51 = uVar34 * -8 + 0x60;
  }
  local_118 = CONCAT44((int)(local_118 >> 0x20),iVar51);
  iVar5 = *(int *)((long)param_3 + 0x2c) + iVar8;
  iVar41 = *(int *)(lVar11 + 0x24);
  if (iVar8 < iVar5 + iVar41 * 2) {
    iVar67 = iVar8;
LAB_013e0cd4:
    do {
      if (*(int *)(lVar11 + 0xc) <= iVar67) break;
      if (iVar67 - iVar8 != 0 && iVar8 <= iVar67) {
        iVar14 = *(int *)(lVar11 + 0x90);
        uVar64 = 0;
        if (iVar14 != 0) {
          uVar64 = iVar67 / iVar14;
        }
        if (iVar67 == uVar64 * iVar14) {
          puVar2 = (undefined1 *)((long)puStack_128 + (long)(0x20 - iVar51 >> 3));
          uVar20 = *(ushort *)(puVar2 + -4);
          uVar21 = *(ushort *)(puVar2 + -2);
          uVar69 = *puVar2;
          uVar19 = puVar2[1];
          FUN_013e3c78(&local_130,0x30);
          if (((ushort)(uVar20 >> 8 | uVar20 << 8) != 0xff20 ||
              (ushort)(uVar21 >> 8 | uVar21 << 8) != 4) || CONCAT11(uVar69,uVar19) != uVar64) {
            uVar36 = FUN_01e78d9c();
            goto LAB_013e2a0c;
          }
          iVar41 = *(int *)(lVar11 + 0x24);
          iVar51 = (uint)local_118;
        }
      }
      iVar14 = 0;
      if (iVar41 != 0) {
        iVar14 = (iVar67 - iVar8) / iVar41;
      }
      lVar43 = *(long *)(lVar77 + 0x18);
      piVar49 = (int *)(lVar43 + (long)iVar14 * 0x2d0);
      pbVar1 = (byte *)((long)puStack_128 + (long)(0x20 - iVar51 >> 3));
      *piVar49 = ((uint)pbVar1[-4] << 0x10 | (uint)pbVar1[-3] << 8 | (uint)pbVar1[-2]) +
                 (*(int *)(lVar11 + 0x94) * 2 + 0x2fU >> 3);
      *(byte *)(piVar49 + 2) = pbVar1[-1];
      *(byte *)((long)piVar49 + 9) = *pbVar1;
      iVar51 = (uint)local_118 + -0x20;
      if ((int)(uint)local_118 < 0x40) {
        if (puStack_128 < local_120) {
          uVar64 = ((uint)*puStack_128 & 0xff00ff00) >> 8 | ((uint)*puStack_128 & 0xff00ff) << 8;
          uVar60 = (ulong)(uVar64 >> 0x10 | uVar64 << 0x10);
        }
        else {
          uVar60 = 0;
        }
        puStack_128 = (ulong *)((long)puStack_128 + 4);
        local_130 = (uVar60 << ((ulong)-(uint)local_118 & 0x3f) | local_130 << 0x20) << 8;
        iVar41 = (uint)local_118 + -8;
        iVar51 = (uint)local_118;
      }
      else {
        local_130 = local_130 << 0x28;
        iVar41 = (uint)local_118 + -0x28;
      }
      if (iVar51 < 0x28) {
        if (puStack_128 < local_120) {
          uVar64 = ((uint)*puStack_128 & 0xff00ff00) >> 8 | ((uint)*puStack_128 & 0xff00ff) << 8;
          uVar60 = (ulong)(uVar64 >> 0x10 | uVar64 << 0x10);
        }
        else {
          uVar60 = 0;
        }
        local_130 = uVar60 << ((ulong)(0x28 - iVar51) & 0x3f) | local_130;
        puStack_128 = (ulong *)((long)puStack_128 + 4);
        iVar41 =
```

## entropy call — FUN_013e0b84

```c
          uVar39 = (ulong)(uVar25 >> 0x10 | uVar25 << 0x10);
                  }
                  else {
                    uVar39 = 0;
                  }
                  local_100 = uVar39 << ((ulong)(uVar59 - 0x20) & 0x3f) | local_100;
                  local_f8 = (ulong *)((long)puVar37 + 0xc);
                  local_e8 = uVar56 * -8 + 0x60;
                }
                iVar53 = *(int *)(lVar11 + 0x90);
                iVar55 = 0;
                if (iVar53 != 0) {
                  iVar55 = iVar8 / iVar53;
                }
                puStack_f0 = puVar54;
                uVar36 = FUN_013e3d84(lVar11,&local_b8,&local_e0,&local_130,&local_100,uVar16,lVar45
                                      ,*(undefined1 *)(lVar63 + lVar45 * 0x38 + 0x40),lVar63 + 0x240
                                      ,lVar63 + 0x288,0,*(undefined8 *)(lVar77 + 0x10),
                                      lVar66 + lVar38 * (ulong)uVar64 * -4,lVar77 + 0x20,
                                      iVar8 == iVar55 * iVar53);
                if (((int)local_120 - *(int *)puVar70) * 8 + local_10c * -4 != iVar58 * 8)
                goto LAB_013e2980;
                if (((int)local_f8 - *(int *)(lVar35 + 0x38)) * 8 - local_e8 !=
                    *(int *)(lVar35 + 0x1c) * 8) goto LAB_013e2988;
                if ((int)uVar36 == 0) {
                  if ((iVar57 - iStack_a0) + ((int)puStack_b0 - (int)puVar50) * 8 != uVar34)
                  goto LAB_013e2990;
                  if (((iVar57 + iVar15 * -8) - (int)local_c8) +
                      ((int)puStack_d8 - (int)puVar50) * 8 != iVar17 * 8) goto LAB_013e2998;
                }
                else if ((int)uVar36 < 0) goto LAB_013e299c;
                lVar65 = lVar65 + 1;
              } while (lVar65 < *(int *)(lVar11 + 0x18));
              lVar65 = -(lVar38 * (ulong)*(uint *)(lVar11 + 0xedc));
            }
            lVar38 = FUN_013e2a2c(*(undefined8 *)(*(long *)(lVar9 + 0x28) + 0xc0),param_2,lVar38,
                                  local_98[lVar38],iVar41,uVar24,lVar66 + lVar65 * 4,lVar47,lVar13,
                                  lVar43);
            lVar43 = lVar43 + lVar38 * 4;
            lVar38 = 1;
            bVar30 = !bVar46;
            bVar33 = false;
            bVar46 = true;
          } while (bVar30);
          lVar38 = *(long *)(*(long *)(*(long *)(lVar9 + 0x28) + 0xc0) + (long)(int)param_2 *
```

## decode output call — FUN_013e0b84

```c
l_158,lVar10,lVar12,lVar42,iVar73);
            if ((int)uVar36 < 0) goto LAB_013e299c;
            iVar73 = iVar73 + 1;
          }
          lVar43 = lVar43 + lVar65 * 4;
          lVar38 = 1;
          bVar33 = false;
        } while (bVar46);
      }
      iVar67 = iVar3;
      if (iVar14 <= iVar3 && iVar5 == iVar14) {
        lVar38 = lVar47;
        if (*(int *)(lVar11 + 0x84) == 0) {
          lVar65 = *(long *)(lVar9 + 0x28);
        }
        else {
          lVar65 = *(long *)(lVar9 + 0x28);
          if (*(int *)((long)&dylib_command_00001008.dylib.current_version + lVar11) == 0) {
            lVar38 = **(long **)(lVar65 + 200) +
                     (ulong)(uint)(*(int *)(lVar11 + 0xedc) * iVar73) * 4;
          }
        }
        FUN_013e2cd8(*(undefined8 *)(lVar65 + 0x20),*(undefined8 *)(lVar65 + 0xc0),param_1,param_2,
                     0xffffffff,0,lVar38);
        uVar36 = FUN_013e2eb8(lVar9,param_2,lVar38,lVar10,lVar12,lVar42,iVar73);
        if ((int)uVar36 < 0) break;
        if (*(int *)(lVar11 + 0x84) == 0) {
          lVar65 = *(long *)(lVar9 + 0x28);
        }
        else {
          lVar65 = *(long *)(lVar9 + 0x28);
          if (*(int *)((long)&dylib_command_00001008.dylib.current_version + lVar11) == 0) {
            lVar38 = **(long **)(lVar65 + 200) +
                     (ulong)(uint)(*(int *)(lVar11 + 0xedc) * (iVar73 + 1)) * 4;
          }
        }
        FUN_013e2cd8(*(undefined8 *)(lVar65 + 0x20),*(undefined8 *)(lVar65 + 0xc0),param_1,param_2,
                     0xffffffff,0,lVar38);
        uVar36 = FUN_013e2eb8(lVar9,param_2,lVar38,lVar10,lVar12,lVar42,iVar73 + 1);
        if ((int)uVar36 < 0) break;
        iVar73 = iVar73 + 2;
        uVar36 = 0;
        if (iVar51 <= iVar3) break;
        goto LAB_013e1808;
      }
      uVar36 = 0;
    } while (iVar3 < iVar51);
  }
  else {
LAB_013e2970:
    uVar36 = 0;
  }
LAB_013e299c:
```

## reconstruction task — FUN_0157e824

```c

void FUN_0157e824(int param_1,ulong param_2,long *param_3)

{
  long lVar1;
  int iVar2;
  int iVar3;
  int iVar4;
  int iVar5;
  long lVar6;
  long lVar7;
  int *piVar8;
  long lVar9;
  long lVar10;

  lVar9 = *(long *)(*param_3 + 0x28);
  piVar8 = (int *)(*(long *)(lVar9 + 0xc0) + (param_2 & 0xffffffff) * 0x48);
  lVar10 = *(long *)(piVar8 + 2);
  if (*(int *)(lVar10 + 0x84) == 0) {
    FUN_01e7a2c0();
  }
  else if (*(int *)((long)&dylib_command_00001008.dylib.current_version + lVar10) == 0) {
    iVar3 = (int)param_3[4];
    iVar5 = *piVar8;
    iVar4 = *(int *)(lVar10 + 0x14);
    iVar2 = iVar4;
    if (7 < iVar4) {
      iVar2 = 8;
    }
    if (iVar5 != 2) {
      iVar2 = iVar4;
    }
    lVar6 = (ulong)(uint)(*(int *)(lVar10 + 0xedc) * iVar3) * 4;
    lVar7 = (ulong)(uint)(*(int *)(lVar10 + 0xecc) << 1) * 4;
    lVar1 = **(long **)(lVar9 + 200) + lVar7 + lVar6;
    lVar6 = (*(long **)(lVar9 + 200))[1] + lVar6;
    (*DAT_08197950)(lVar1,lVar6,lVar6 + lVar7,
                    lVar1 + (ulong)(uint)(*(int *)(lVar10 + 0xed0) << 1) * 4,param_2,param_3[1],
                    param_3[2],param_3[3],iVar3,*(int *)(lVar10 + 0xedc),
                    *(int *)((long)param_3 + 0x24),*(int *)(lVar10 + 8) << 1,
                    *(int *)(lVar10 + 0xc) <= *(int *)((long)param_3 + 0x24) + iVar3,lVar9 + 0xd0,
                    lVar10 + 0x40,*(int *)(lVar10 + 0x2c) - iVar2,iVar5,
                    *(undefined8 *)(*(long *)(*(long *)(lVar9 + 0x20) + (long)param_1 * 8) + 0xa0));
    return;
  }
                    /* WARNING: Subroutine does not return */
  FUN_01e7a2e8();
}
```

## reconstruction thunk — FUN_013c951c

```c

void FUN_013c951c(void)

{
  DAT_08197950 = FUN_013c8bc8;
  FUN_013c8bc8();
  return;
}
```

## acquire callback — FUN_013c8bc8

```c
m_10 & 0xffffffff) << 2;
    uVar40 = -(ulong)(param_9._4_4_ >> 0x1f) & 0xfffffffc00000000 | (ulong)param_9._4_4_ << 2;
    lVar35 = param_3 - uVar38;
    lVar32 = param_4;
    lVar42 = param_2;
    lVar52 = param_1;
    lVar53 = param_3;
    do {
      iVar2 = (int)uVar34 + (int)param_9;
      uVar9 = iVar2 * 2;
      iVar2 = param_11._4_4_ + iVar2 * -2;
      if (1 < iVar2) {
        iVar2 = 2;
      }
      lVar3 = uVar34 + (param_9 & 0xffffffff);
      if ((int)uVar9 < param_11._4_4_) {
        local_e0 = param_18;
        if ((param_17 != 5) ||
           (local_e0 = (undefined8 *)
                       (*param_6)(param_5,0,lVar3,
                                  -(ulong)((uint)(iVar2 * (int)param_12) >> 0x1f) &
                                  0xfffffffe00000000 | (ulong)(uint)(iVar2 * (int)param_12) << 1,
                                  param_8), local_e0 != (undefined8 *)0x0)) {
          puVar29 = (undefined8 *)((long)local_e0 + ((ulong)(long)(int)param_12 >> 1) * 4);
          if (param_11._4_4_ <= (int)(uVar9 | 1)) {
            puVar29 = (undefined8 *)0x0;
          }
          lVar43 = uVar34 * lVar37 * 4;
          puVar41 = (undefined4 *)(param_2 + lVar43);
          puVar47 = puVar41;
          if (param_13 != 0) goto LAB_013c8e60;
          goto LAB_013c8eb4;
        }
LAB_013c94cc:
        uVar30 = 0xffff5
```

## scalar / 16-bit output — FUN_013c8bc8

```c
= 0)];
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
                uVar31 = *(uint *)(*(long *)(param_14 + 2) + (long)(int)uVar
```

## completion callback — FUN_013c8bc8

```c
*param_15 != '\0') {
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
              *(short *)((long)puVar29 + 2) = (short)uVar4;
              puVar29 = (undefined8 *)((long)puVar29 + 4);
            }
            lVar51 = lVar51 + 4;
            iVar50 = iVar50 + -1;
            iVar54 = iVar76;
            iVar75 = iVar62;
          } while (iVar50 != 0);
        }
      }
      if ((int)uVar9 < param_11._4_4_) {
        if (param_17 == 2) {
          uVar33 = (ulong)(iVar2 * (int)param_12);
          puVar29 = (undefined8 *)(*param_6)(param_5,0,lVar3,uVar33,param_8);
          if (local_e0 == (undefined8 *)0x0) goto LAB_013c94cc;
          (*DAT_08198578)(puVar29,param_18,uVar33);
          if (param_7 != (code *)0x0) goto LAB_013c8d1c;
        }
        else {
          if (param_17 != 5) {
            FUN_01e78cd4();
            DAT_08197948 = FUN_013c86b4;
            FUN_013c86b4();
            return;
          }
          if (param_7 != (code *)0x0) {
            uVar33 = -(ulong)((uint)(iVar2 * (int)param_12) >> 0x1f) & 0xfffffffe00000000 |
                     (ulong)(uint)(iVar2 * (int)param_12) << 1;
            puVar29 = local_e0;
LAB_013c8d1c:
            (*param_7)(param_5,0,lVar3,uVar33,puVar29,param_8);
          }
        }
      }
      uVar34 = uVar34 + 1;
      lVar42 = lVar42 + uVar38;
      lVar53 = lVar53 + uVar38;
      lVar52 = lVar52 + uVar40;
      lVar32 = lVar32 + uVar40;
      lVar35 = lVar35 + uVar38;
    } while (uVar34 != (param_11 & 0xffffffff));
  }
  uVar30 = 0;
  lVar42 = *(long *)PTR____stack_chk_guard_07a703a0;
joined_r0x013c94e4:
  if (lVar42 == lVar36) {
    return;
  }
                    /* WARNING: Subroutine does not return */
  ___stack_chk_fail(uVar30);
}
```

## unary — FUN_013e5828

```c

int FUN_013e5828(ulong *param_1)

{
  uint *puVar1;
  uint uVar2;
  int iVar3;
  int iVar4;
  ulong uVar5;
  ulong uVar6;
  ulong uVar7;

  uVar5 = *param_1;
  if (-1 < (long)uVar5) {
    *param_1 = uVar5 << 1;
    iVar4 = (int)param_1[3];
    *(int *)(param_1 + 3) = iVar4 + -1;
    if (0x20 < iVar4) {
      return 0;
    }
    puVar1 = (uint *)param_1[1];
    if (puVar1 < (uint *)param_1[2]) {
      uVar2 = (*puVar1 & 0xff00ff00) >> 8 | (*puVar1 & 0xff00ff) << 8;
      uVar6 = (ulong)(uVar2 >> 0x10 | uVar2 << 0x10);
    }
    else {
      uVar6 = 0;
    }
    *param_1 = uVar6 << ((ulong)(0x21 - iVar4) & 0x3f) | uVar5 << 1;
    param_1[1] = (ulong)(puVar1 + 1);
    *(int *)(param_1 + 3) = iVar4 + 0x1f;
    return 0;
  }
  if ((uVar5 >> 0x3e & 1) == 0) {
    *param_1 = uVar5 << 2;
    iVar4 = (int)param_1[3];
    *(int *)(param_1 + 3) = iVar4 + -2;
    if (0x21 < iVar4) {
      return 1;
    }
    puVar1 = (uint *)param_1[1];
    if (puVar1 < (uint *)param_1[2]) {
      uVar2 = (*puVar1 & 0xff00ff00) >> 8 | (*puVar1 & 0xff00ff) << 8;
      uVar6 = (ulong)(uVar2 >> 0x10 | uVar2 << 0x10);
    }
    else {
      uVar6 = 0;
    }
    *param_1 = uVar6 << ((ulong)(0x22 - iVar4) & 0x3f) | uVar5 << 2;
    param_1[1] = (ulong)(puVar1 + 1);
    *(int *)(param_1 + 3) = iVar4 + 0x1e;
    return 1;
  }
  if ((uVar5 >> 0x3d & 1) == 0) {
    *param_1 = uVar5 << 3;
    iVar4 = (int)param_1[3];
    *(int *)(param_1 + 3) = iVar4 + -3;
    if (0x22 < iVar4) {
      return 2;
    }
    puVar1 = (uint *)param_1[1];
    if (puVar1 < (uint *)param_1[2]) {
      uVar2 = (*puVar1 & 0xff00ff00) >> 8 | (*puVar1 & 0xff00ff) << 8;
      uVar6 = (ulong)(uVar2 >> 0x10 | uVar2 << 0x10);
    }
    else {
      uVar6 = 0;
    }
    *param_1 = uVar6 << ((ulong)(0x23 - iVar4) & 0x3f) | uVar5 << 3;
    param_1[1] = (ulong)(puVar1 + 1);
    *(int *)(param_1 + 3) = iVar4 + 0x1d;
    return 2;
  }
  iVar4 = 0;
  do {
    uVar2 = iVar4 + 0x3c;
    iVar4 = iVar4 + -1;
  } while ((uVar5 >> ((ulong)uVar2 & 0x3f) & 1) != 0);
  uVar5 = uVar5 << ((ulong)(3 - iVar4) & 0x3f);
  *param_1 = uVar5;
  uVar6 = param_1[3];
  iVar3 = (int)uVar6 + iVar4 + -3;
  *(int *)(param_1 + 3) = iVar3;
  if (0x1f < iVar3) {
    return 2 - iVar4;
  }
  puVar1 = (uint *)param_1[1];
  if (puVar1 < (uint *)param_1[2]) {
    uVar2 = (*puVar1 & 0xff00ff00) >> 8 | (*puVar1 & 0xff00ff) << 8;
    uVar7 = (ulong)(uVar2 >> 0x10 | uVar2 << 0x10);
  }
  else {
    uVar7 = 0;
  }
  iVar3 = (int)uVar6 + iVar4;
  *param_1 = uVar7 << ((ulong)(0x23 - iVar3) & 0x3f) | uVar5;
  param_1[1] = (ulong)(puVar1 + 1);
  *(int *)(param_1 + 3) = iVar3 + 0x1d;
  return 2 - iVar4;
}
```

## Unary usage — FUN_013e3d84

```c
 }
          }
        }
        else {
          uVar37 = 0;
          if (7 < (int)uVar7) {
            do {
              while( true ) {
                uVar58 = uVar29 + 0x1f;
                if (-1 < (int)uVar29) {
                  uVar58 = uVar29;
                }
                pbVar28 = pbVar23 + uVar37;
                if ((local_a0[(int)uVar58 >> 5] >> (ulong)(uVar29 & 0x1f) & 1) == 0) break;
                *(ulong *)pbVar28 = uVar34 * 0x101010101010101;
                uVar37 = uVar37 + 8;
                uVar29 = uVar29 - 1;
                if ((int)(uVar7 - 7) <= (int)uVar37) goto LAB_013e54a0;
              }
              cVar20 = FUN_013e5828(param_3);
              *pbVar28 = bVar25 + cVar20;
              cVar20 = FUN_013e5828(param_3);
              pbVar28[1] = bVar25 + cVar20;
              cVar20 = FUN_013e5828(param_3);
              pbVar28[2] = bVar25 + cVar20;
              cVar20 = FUN_013e5828(param_3);
              pbVar28[3] = bVar25 + cVar20;
              cVar20 = FUN_013e5828(param_3);
              pbVar28[4] = bVar25 + cVar20;
              cVar20 = FUN_013e5828(param_3);
              pbVar28[5] = bVar25 + cVar20;
              cVar20 = FUN_013e5828(param_3);
              pbVar28[6] = bVar25 + cVar20;
              cVar20 = FUN_013e5828(param_3);
              pbVar28[7] = bVar25 + cVar20;
              uVar37 = uVar37 + 8;
              uVar29 = uVar29 - 1;
            } while ((int)uVar37 < (int)(uVar7 - 7));
          }
LAB_013e54a0:
          uVar58 = uVar7 & 7;
          if (-1 < (int)-uVar7) {
            uVar58 = -(-uVar7 & 7);
          }
          uVar27 = (ulong)uVar58;
          if (uVar58 != 0) {
            if (((byte)local_a0[0] & 1) == 0) {
              pbVar23 = pbVar23 + (uVar37 & 0xffffffff);
              if (0 < (int)uVar58) {
                do {
                  cVar20 = FUN_013e5828(param_3);
                  *pbVar23 = bVar25 + cVar20;
                  uVar27 = uVar27 - 1;
                  pbVar23 = pbVar23 + 1;
                } while (uVar27 != 0);
              }
            }
            else {
              _memset(pbVar23 + (uVar37 & 0xffffffff),uVar26,(long)(int)uVar58);
            }
          }
        }
        *(undefined1 *)(param_14 + lVar41) = 1;
joined_r0x013e4e2c:
        if (0 < (int)uVar7) {
          iVar21 = uVar32 - 0x10;
          auVar79._4_4_ = iVar21;
          auVar79._0_4_ = iVar21;
          auVar79._8_4_ = iVar21;
          auVar79._12_4_ = iVar21;
          iVar21 = uVar24 << 2;
          do {
            bVar25 = *(byte *)(param_12 + lVar39);
```

## worker BLR

```asm
014d5ad0 str w10,[x9, x25#0x0]
014d5ad4 ldrb w9,[sp, #0x18]
014d5ad8 cbz w9,0x014d5db8
014d5adc ldr x0,[sp, #0x10]
014d5ae0 bl 0x01e7f338
014d5ae4 strb wzr,[sp, #0x18]
014d5ae8 ldr x0,[sp]
014d5aec bl 0x01e80028
014d5af0 mov x8,x0
014d5af4 cbnz w0,0x014d5b14
014d5af8 ldr x8,[x19, #0x28]
014d5afc ldr x8,[x8, x27, LSL #0x3]
014d5b00 ldp x9,x2,[x8]
014d5b04 ldr w0,[sp, #0xc]
014d5b08 mov x1,x23
014d5b0c blr x9
014d5b10 mov x8,x0
014d5b14 ldr x0,[sp, #0x10]
014d5b18 cbz x0,0x014d5da0
014d5b1c mov x23,x8
014d5b20 ldrb w8,[sp, #0x18]
014d5b24 cbnz w8,0x014d5dac
014d5b28 bl 0x01e7f32c
014d5b2c mov w8,#0x0
014d5b30 mov w9,#0x1
014d5b34 strb w9,[sp, #0x18]
014d5b38 ldr x9,[x19, #0x28]
014d5b3c ldr x9,[x9, x27, LSL #0x3]
```

## acquire callback BLR

```asm
013c8d90 add x8,x4,x10
013c8d94 stur x8,[x29, #-0xc0]
013c8d98 cmp w9,w12
013c8d9c stp x3,x5,[x29, #-0xa8]
013c8da0 stur w12,[x29, #-0x98]
013c8da4 b.le 0x013c8e40
013c8da8 ldr x0,[sp, #0x40]
013c8dac ldr w8,[sp, #0xcc]
013c8db0 cmp w8,#0x5
013c8db4 b.ne 0x013c8e0c
013c8db8 ldr x8,[sp, #0x38]
013c8dbc ldur w9,[x29, #-0xc4]
013c8dc0 mul w8,w9,w8
013c8dc4 sbfiz x3,x8,#0x1,#0x20
013c8dc8 ldr w0,[sp, #0xa4]
013c8dcc mov w1,#0x0
013c8dd0 ldur x2,[x29, #-0xc0]
013c8dd4 mov x20,x4
013c8dd8 ldr x4,[sp, #0xa8]
013c8ddc ldr x8,[sp, #0x30]
013c8de0 blr x8
013c8de4 ldur w12,[x29, #-0x98]
013c8de8 ldp x3,x5,[x29, #-0xa8]
013c8dec movi v23.2D,#0x0
013c8df0 mvni v22.4S,#0x1
013c8df4 ldp q21,q20,[sp, #0x50]
013c8df8 ldp q19,q18,[sp, #0x70]
013c8dfc mov x4,x20
013c8e00 ldr x1,[sp, #0xe0]
013c8e04 ldur x2,[x29, #-0xd8]
```

## transform dispatch BLR

```asm
013c8c90 csel w25,w11,wzr,gt
013c8c94 dup v19.4S,w25
013c8c98 neg w11,w22
013c8c9c sub w12,w8,#0x1
013c8ca0 mov w13,#0x2
013c8ca4 lsl w12,w13,w12
013c8ca8 sub w12,w12,#0x1
013c8cac cmp w8,#0x0
013c8cb0 csinv w27,w12,wzr,ne
013c8cb4 dup v20.4S,w11
013c8cb8 dup v21.4S,w27
013c8cbc stur w15,[x29, #-0xac]
013c8cc0 sub w8,w15,#0x5
013c8cc4 and w8,w8,#0xfffffffc
013c8cc8 add w8,w8,#0x4
013c8ccc str w8,[sp, #0x18]
013c8cd0 sxtw x8,w9
013c8cd4 stur x8,[x29, #-0xb8]
013c8cd8 sxtw x8,w10
013c8cdc str x8,[sp, #0x10]
013c8ce0 sbfiz x8,x9,#0x2,#0x20
013c8ce4 sbfiz x9,x10,#0x2,#0x20
013c8ce8 stp x9,x8,[sp, #0xb0]
013c8cec sub x5,x2,x8
013c8cf0 mvni v22.4S,#0x1
013c8cf4 movi v23.2D,#0x0
013c8cf8 stp x0,x3,[sp, #0x20]
013c8cfc mov x24,x0
013c8d00 mov x26,x2
013c8d04 mov x23,x1
013c8d08 stur x2,[x29, #-0xd8]
013c8d0c str x1,[sp, #0xe0]
013c8d10 stp q19,q18,[sp, #0x70]
013c8d14 stp q21,q20,[sp, #0x50]
013c8d18 b 0x013c8d6c
013c8d1c ldr x5,[sp, #0xa8]
013c8d20 ldr x8,[sp, #0x98]
013c8d24 blr x8
013c8d28 movi v23.2D,#0x0
013c8d2c mvni v22.4S,#0x1
013c8d30 ldp q21,q20,[sp, #0x50]
013c8d34 ldp q19,q18,[sp, #0x70]
013c8d38 ldur x4,[x29, #-0x90]
013c8d3c add x4,x4,#0x1
013c8d40 ldp x9,x8,[sp, #0xb0]
013c8d44 add x23,x23,x8
013c8d48 add x26,x26,x8
013c8d4c add x24,x24,x9
013c8d50 ldp x3,x5,[x29, #-0xa8]
013c8d54 add x3,x3,x9
013c8d58 add x5,x5,x8
013c8d5c ldp x8,x1,[sp, #0xd8]
013c8d60 cmp x4,x8
013c8d64 ldur x2,[x29, #-0xd8]
013c8d68 b.eq 0x013c9490
013c8d6c ldr x10,[sp, #0xd0]
013c8d70 add w8,w4,w10
013c8d74 lsl w12,w8,#0x1
013c8d78 ldur w9,[x29, #-0x84]
013c8d7c subs w8,w9,w12
013c8d80 cmp w8,#0x2
013c8d84 mov w11,#0x2
013c8d88 csel w8,w8,w11,lt
013c8d8c stur w8,[x29, #-0xc4]
013c8d90 add x8,x4,x10
```

## unary bit tests

```asm
013e5828 ldr x9,[x0]
013e582c tbnz x9,#0x3f,0x013e5854
013e5830 lsl x10,x9,#0x1
013e5834 str x10,[x0]
013e5838 ldr w9,[x0, #0x18]
013e583c sub w8,w9,#0x1
013e5840 str w8,[x0, #0x18]
013e5844 cmp w9,#0x20
013e5848 b.le 0x013e587c
013e584c mov w0,#0x0
013e5850 ret
013e5854 tbnz x9,#0x3e,0x013e58b8
013e5858 lsl x9,x9,#0x2
013e585c str x9,[x0]
013e5860 ldr w8,[x0, #0x18]
013e5864 sub w10,w8,#0x2
013e5868 str w10,[x0, #0x18]
013e586c cmp w8,#0x21
013e5870 b.le 0x013e58e0
013e5874 mov w0,#0x1
013e5878 ret
013e587c ldp x11,x8,[x0, #0x8]
013e5880 cmp x11,x8
013e5884 b.cs 0x013e591c
013e5888 ldr w8,[x11]
013e588c rev w12,w8
013e5890 mov w13,#0x21
013e5894 sub w13,w13,w9
013e5898 lsl x12,x12,x13
013e589c orr x10,x12,x10
013e58a0 add x11,x11,#0x4
013e58a4 stp x10,x11,[x0]
013e58a8 add w9,w9,#0x1f
013e58ac str w9,[x0, #0x18]
013e58b0 mov w0,#0x0
013e58b4 ret
013e58b8 tbnz x9,#0x3d,0x013e5924
013e58bc lsl x9,x9,#0x3
013e58c0 str x9,[x0]
013e58c4 ldr w8,[x0, #0x18]
013e58c8 sub w10,w8,#0x3
013e58cc str w10,[x0, #0x18]
013e58d0 cmp w8,#0x22
013e58d4 b.le 0x013e596c
013e58d8 mov w0,#0x2
013e58dc ret
013e58e0 ldp x10,x11,[x0, #0x8]
013e58e4 cmp x10,x11
013e58e8 b.cs 0x013e59a8
013e58ec ldr w11,[x10]
013e58f0 rev w11,w11
013e58f4 mov w12,#0x22
013e58f8 sub w12,w12,w8
013e58fc lsl x11,x11,x12
013e5900 orr x9,x11,x9
013e5904 add x10,x10,#0x4
013e5908 stp x9,x10,[x0]
013e590c add w8,w8,#0x1e
013e5910 str w8,[x0, #0x18]
013e5914 mov w0,#0x1
013e5918 ret
013e591c mov x12,#0x0
013e5920 b 0x013e5890
013e5924 mov w8,#0x0
013e5928 add w10,w8,#0x3c
```


