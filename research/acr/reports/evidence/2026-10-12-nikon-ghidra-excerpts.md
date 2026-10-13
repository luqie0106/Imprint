# ACR 17.2 ARM64 — bounded Ghidra-project evidence

2026-10-12. Read-only MCP queries against `/Camera_Raw_arm64.analysis-383152`. Pseudocode is recovered, non-source, and contains inaccurate prototypes and data labels. Excerpts only; NOT complete large-function coverage. Character slices below refer to this session's decompile response and are not VM addresses.

## FUN_012da138 entry / selected TIFF fields

```c
uVar29 = auVar56._8_8_;
  plVar28 = auVar56._0_8_;
  local_88 = *(long *)PTR____stack_chk_guard_07a703a0;
  if (*(int *)(param_5 + 0x1c) != 0x2a) {
LAB_012da33c:
    plVar48 = (long *)0x0;
    goto LAB_012da340;
  }
  lVar42 = *(long *)(param_5 + 0x28);
  if (0x60 < *(int *)(lVar42 + 0xcc8) - 300U) goto LAB_012da33c;
  lVar45 = *(long *)(param_5 + 0x60);
  uVar39 = *(long *)(param_5 + 0x68) - lVar45;
  if ((uVar39 & 0x7fffffff8) == 0) goto LAB_012da33c;
  uVar37 = 0;
  iVar26 = -1;
  do {
    iVar4 = (int)uVar37;
    if (*(int *)(*(long *)(lVar45 + uVar37 * 8) + 0xc) != 0) {
      iVar4 = iVar26;
    }
    uVar37 = uVar37 + 1;
    iVar26 = iVar4;
  } while ((uVar39 >> 3 & 0xffffffff) != uVar37);
  if (iVar4 == -1) goto LAB_012da33c;
  plVar43 = *(long **)(lVar45 + (long)iVar4 * 8);
  iVar26 = *(int *)((long)plVar43 + 0x34);
  if (((iVar26 != 0x8023 && iVar26 != 2) || ((*(uint *)((long)plVar43 + 0x54) | 2) != 3)) ||
     ((*(uint *)(plVar43 + 3) | 2) != 0xe)) goto LAB_012da33c;
  bVar17 = iVar26 == 2;
  bVar18 = *(uint *)((long)plVar43 + 0x54) == 3;
  bVar19 = *(uint *)(plVar43 + 3) == 0xe;
  bVar14 = (bVar17 && bVar18) && bVar19;
  uVar24 = *(int *)(lVar42 + 0xcc8) - 0x132;
  if ((uVar24 < 8) && ((1 << (ulong)(uVar24 & 0x1f) & 0xc3U) != 0)) {
    *(undefined4 *)((long)plVar43 + 0x2c) = 1;
  }
  plVar27 = (long *)(**(code **)(*plVar28 + 0x68))(plVar28);
  FUN_0126fcc8(plVar27 + 0x9c,"image/x-nikon-nef");
  FUN_00af1b94(plVar27,param_5);
  FUN_00aa4cbc(plVar28,plVar27,*(undefined4 *)(lVar42 + 0xcc8));
  local_1f68 = 0;
  uStack_1f64 = 0;
  puVar47 = (uint *)((long)plVar43 + 0x14);
  local_1f60 = *puVar47;
  iStack_1f5c = (int)plVar43[2];
  uVar6 = *(ushort *)(&DAT_000014b2 + lVar42);
  uVar7 = *(ushort *)(&DAT_000014b6 + lVar42);
  if (((DAT_084e4f50 & 1) == 0) && (iVar26 = ___cxa_guard_acquire(&DAT_084e4f50), iVar26 != 0)) {
    param_2 = 0.18;
    DAT_084e4f48 = (ushort)(int)((double)(int)((uint)uVar7 - (uint)uVar6) * 0.18);
    ___cxa_guard_release(&DAT_084e4f50);
  }
  bVar20 = *(int *)((long)plVar43 + 0x2c) == 0x8799;
  iVar26 = *(int *)(lVar42 + 0xcc8);
  plVar41 = plVar27;
  if (0x60 < iVar26 - 300U) goto LAB_012dbfc0;
  iVar4 = *(int *)(&DAT_000014a8 + lVar42);
  switch(iVar26) {
  case 300:
  case 0x12e:
```

## FUN_012da138 scalar and IPX dispatch

```c
N_0126b0e0(dVar52 + -0.3113974273951928,plVar27 + 0x27,100);
      local_ff0._0_8_ = (undefined1 *)0x0;
      dVar52 = (double)FUN_0126b1dc(plVar27 + 0x2d);
      FUN_0126b200(dVar52 / 0.8058608058608059,local_ff0,0);
      FUN_0130cb4c(plVar27,local_ff0);
    }
    uVar39 = FUN_00e130d4(plVar28,plVar27);
    if ((uVar39 & 1) != 0) goto LAB_012da340;
    local_ff0 = FUN_00c1c7b8(plVar43);
    uVar23 = 3;
    if ((!bVar17 || !bVar18) || !bVar19) {
      uVar23 = 1;
    }
    local_1f70 = (long *)(**(code **)(*plVar28 + 0x70))(plVar28,local_ff0,uVar23,3);
    if (bVar20) {
      if ((int)local_1f60 < (int)local_1f70[2]) {
        local_ff0._0_8_ = local_1f70[1];
        local_ff0._12_4_ = (undefined4)((ulong)local_1f70[2] >> 0x20);
        local_ff0._8_4_ = local_1f60;
        (**(code **)(*local_1f70 + 0x28))(local_1f70,local_ff0);
      }
      FUN_00ae2d38(plVar27,0,0);
      lVar33 = plVar43[0x12];
      FUN_0126b048(local_1f58,uVar29,
                   "/System/Library/Frameworks/ColorSync.framework/Versions/A/ColorSync"
                   [lVar42 + 0x20]);
      FUN_00dd5b68(local_ff0,"ReadNikonCompressed_Scalar",0);
      FUN_012e001c(plVar28,uVar29,lVar45,lVar33,local_2000._4_4_,local_2000 & 0xffffffff,iVar26,
                   local_1f70,extraout_w1);
      FUN_00dd5bc4(local_ff0);
      FUN_0126b080(local_1f58);
      goto LAB_012ddfa4;
    }
    if (iVar4 - 0xdU < 2) {
      FUN_012e0504(plVar28,(&DAT_000014a4)[lVar42],uVar29,plVar43[0x12],plVar43[0x34]);
      goto LAB_012ddfa4;
    }
    if (bVar21) {
      FUN_012691f0(uVar29,plVar43[0x12]);
      FUN_00c280e8
```

## FUN_012e0504 IPX lifecycle wrapper

```c
void FUN_012e0504(undefined8 param_1,char param_2,undefined8 param_3,undefined8 param_4,
                 undefined8 param_5,undefined8 param_6)

{
  undefined *local_78;
  undefined8 local_70;
  undefined8 uStack_68;
  undefined8 uStack_60;
  undefined8 uStack_58;
  undefined8 local_50;
  undefined4 local_48;

  if (param_2 != '\0') {
    local_78 = &DAT_080a33a0;
    uStack_68 = 0;
    local_70 = 0;
    uStack_58 = 0;
    uStack_60 = 0;
    local_50 = 0;
    local_48 = 0xffffffff;
    FUN_012d9160(&local_78);
    FUN_012d92d0(&local_78,param_1,param_3,param_4,param_5,param_6);
    FUN_012d90d0(&local_78);
  }
  return;
}
```

## FUN_012d92d0 input / geometry / output

```c
0xffffffff;
  if (*(char *)((long)param_3 + 9) == '\0') {
    lVar14 = (**(code **)*param_3)(param_3);
    param_3[2] = lVar14;
    *(undefined1 *)((long)param_3 + 9) = 1;
    uVar9 = lVar14 - param_3[4];
  }
  else {
    uVar9 = param_3[2] - param_3[4];
  }
  if (uVar9 < uVar11) {
                    /* WARNING: Subroutine does not return */
    FUN_012e9c48(100000,0,"dataBytes too large",0);
  }
  plVar3 = (long *)(**(code **)(*param_2 + 0x10))(param_2,param_5);
  uVar4 = FUN_01395ad4(param_2);
  thunk_FUN_0126ad7c(auStack_80,param_3,uVar4,param_4,uVar11);
  lVar14 = plVar3[1];
  lVar12 = plVar3[2];
  FUN_01269264(param_3,lVar12,(int)lVar14,0);
  iVar2 = FUN_013b6864(*(undefined8 *)(param_1 + 0x10),(int)lVar14,lVar12,
                       *(undefined8 *)(param_1 + 0x18));
  *(int *)(param_1 + 0x30) = iVar2;
  if (iVar2 != 0) {
LAB_012d9648:
                    /* WARNING: Subroutine does not return */
    FUN_012e9c48(0x186a6,0,"DecodeTiles failed",0);
  }
  iVar2 = FUN_013b6264(*(undefined8 *)(param_1 + 0x10),*(undefined8 *)(param_1 + 0x20),
                       *(undefined8 *)(param_1 + 0x28),*(undefined8 *)(param_1 + 0x18),
                       *(undefined8 *)(param_1 + 0x18));
  *(int *)(param_1 + 0x30) = iVar2;
  if (iVar2 != 0) goto LAB_012d9648;
  uVar4 = FUN_01404820(*(undefined8 *)(param_1 + 0x18));
  uVar5 = FUN_01404828(*(undefined8 *)(param_1 + 0x18));
  uVar13 = 100000;
  uVar6 = 0x186a6;
  pcVar8 = "Invalid image geometry";
  if ((10000 < (uint)uVar4) || (7000 < (uint)uVar5)) goto LAB_012d9700;
  if (*(int *)(param_6 + 0xc) <= *(int *)(param_6 + 0x14)) {
    iVar2 = FUN_0138661c(*(int *)(param_6 + 0x14),*(int *)(param_6 + 0xc),local_d8);
    if (iVar2 != 0) {
      if (local_d8[0] <= (uint)uVar4) goto LAB_012d942c;
LAB_012d96a4:
      uVar6 = 0x186a6;
      pcVar8 = "Invalid image geometry";
      goto LAB_012d9700;
    }
    pcVar8 = "Overflow computing rectangle width";
LAB_012d96f8:
    uVar6 = 100000;
LAB_012d9700:
                    /* WARNING: Subroutine does not return */
    FUN_012e9c48(uVar6,0,pcVar8,0);
  }
LAB_012d942c:
  if (*(int *)(param_6 + 8) <= *(int *)(param_6 + 0x10)) {
    iVar2 = FUN_0138661c(*(int *)(param_6 + 0x10),*(int *)(param_6 + 8),local_d8);
    if (iVar2 == 0) {
      pcVar8 = "Overflow computing rectangle height";
      goto LAB_012d96f8;
    }
    if ((uint)uVar5 < local_d8[0]) goto LAB_012d96a4;
  }
  uVar9 = FUN_0144f07c(*(undefined8 *)(param_1 + 0x18),*(undefined8 *)(param_1 + 0x20));
  if (*(int *)(param_6 + 0x14) < *(int *)(param_6 + 0xc)) {
    iVar2 = *(int *)(param_6 + 0x10);
    iVar1 = *(int *)(param_6 + 8);
    uVar10 = 0;
    if (iVar2 < iVar1) goto LAB_012d9484;
LAB_012d94ac:
    iVar2 = FUN_0138661c(iVar2,iVar1,local_d8);
    if (iVar2 == 0) {
      pcVar8 = "Overflow computing rectangle height";
      goto LAB_012d9724;
    }
  }
  else {
    iVar2 = FUN_0138661c(*(int *)(param_6 + 0x14),*(int *)(param_6 + 0xc),local_d8);
    if (iVar2 == 0) {
      pcVar8 = "Overflow computing rectangle width";
      goto LAB_012d9724;
    }
    iVar2 = *(int *)(param_6 + 0x10);
    iVar1 = *(int *)(param_6 + 8);
    uVar10 = local_d8[0];
    if (iVar1 <= iVar2) goto LAB_012d94ac;
LAB_012d9484:
    local_d8[0] = 0;
  }
  if (uVar9 <= uVar10 * local_d8[0] * 2) {
    plVar7 = (long *)(**(code **)(*param_2 + 0x10))(param_2,uVar9);
    lVar14 = plVar7[2];
    iVar2 = FUN_013b68d0(*(undefined8 *)(param_1 + 0x10),*(undefined8 *)(param_1 + 0x28),
                         *(undefined8 *)(param_1 + 0x20),uVar11,lVar12,uVar9,lVar14,0,0);
    *(int *)(param_1 + 0x30) = iVar2;
    if (iVar2 != 0) {
                    /* WARNING: Subroutine does not return */
      FUN_012e9c48(0x186a6,0,"DecodeTiles failed",0);
    }
    local_90 = 0;
    iVar2 = FUN_01386a34(uVar5,&uStack_88);
    if ((iVar2 != 0) && (uVar11 = FUN_01386a34(uVar4,(long)&uStack_88 + 4), (uVar11 & 1) != 0)) {
      FUN_01300c7c(local_d8);
      uStack_c8 = uStack_88;
      uVar5 = uStack_c8;
      local_d0 = local_90;
      uVar4 = local_d0;
      local_c0 = 0x100000000;
      local_b4 = 1;
      uStack_c8._4_4_ = (int)((ulong)uStack_88 >> 0x20);
      iVar1 = uStack_c8._4_4_;
      local_d0._4_4_ = (int)((ulong)local_90 >> 0x20);
      iVar2 = local_d0._4_4_;
      local_d0 = uVar4;
      uStack_c8 = uVar5;
      if (uStack_c8._4_4_ < local_d0._4_4_) {
        local_64 = 0;
      }
      else {
        uVar11 = FUN_0138661c(iVar1,iVar2,&local_64);
        if ((uVar11 & 1) == 0) {
                    /* WARNING: Subroutine does not return */
          FUN_012e9c48(100000,0,"Overflow computing rectangle width",0);
        }
      }
      local_ac = 0x200000003;
      local_b8 = local_64;
      local_a0 = lVar14;
      FUN_0127eca8(param_6,local_d8);
      FUN_01301098(local_d8);
      (**(code **)(*plVar7 + 8))(plVar7);
      thunk_FUN_0126af40(auStack_80);
      (**(code **)(*plVar3 + 8))(plVar3);
      return;
    }
                    /* WARNING:
```

## FUN_013b62dc header parser call and geometry

```c
50) = 0;
    *(undefined8 *)(lVar13 + 0x48) = 0;
    *(undefined8 *)(lVar13 + 0x60) = 0;
    *(undefined8 *)(lVar13 + 0x58) = 0;
    *(undefined8 *)(lVar13 + 0x70) = 0;
    *(undefined8 *)(lVar13 + 0x68) = 0;
    *(undefined8 *)(lVar13 + 0x80) = 0;
    *(undefined8 *)(lVar13 + 0x78) = 0;
    uVar9 = FUN_013d3268(plVar5,pvVar7,param_3);
    uVar3 = (uint)uVar9;
    if ((int)uVar3 < 0) {
      return uVar9;
    }
    sVar10 = (size_t)(int)uVar3;
    if (param_3 < (long)sVar10) {
      return 0xffff5fc0;
    }
    lVar6 = *plVar5;
    if (uVar3 < 0xe9) {
      _memcpy(*(void **)(param_1 + 0x10),pvVar7,sVar10);
      uVar9 = FUN_013d5810(param_1 + 0x20,lVar6);
      if ((int)uVar9 != 0) {
        return uVar9;
      }
      *(size_t *)(param_1 + 0x18) = sVar10;
    }
  }
  else {
    uVar9 = FUN_013d5810(plVar5,*(undefined8 *)(param_1 + 0x20));
    if ((int)uVar9 != 0) {
      return uVar9;
    }
    lVar6 = *plVar5;
  }
  *(size_t *)(lVar11 + uVar12 * 0x48 + 0x28) = sVar10;
  iVar2 = *(int *)(lVar6 + 8);
  if (*(int *)(lVar6 + 0x84) == 0) {
    iVar1 = *(int *)(lVar6 + 0xc);
    uVar4 = *(undefined4 *)(lVar6 + 0x18);
  }
  else {
    iVar2 = iVar2 << 1;
    iVar1 = *(int *)(lVar6 + 0xc) * 2 - *(int *)(lVar6 + 0x10);
    uVar4 = 1;
  }
  if (param_6 != 0) {
    FUN_01404804(param_6,iVar2,iVar1);
    FUN_01404830(param_6,*(undefined4 *)(lVar6 + 0x14));
    FUN_0140484c(param_6,*(byte *)(lVar6 + 0x34) & 1);
    FUN_01404858(param_6,uVar4);
    FUN_01404884(param_6,0);
    FUN_014048a4(param_6,*(int *)(lVar6 + 0x84) != 0);
    lVar8 = 0x70;
    if (*(int *)(lVar6 + 0x44) != 3) {
      lVar8 = 0x58;
    }
    FUN_014048c4(param_6,*(int *)(lVar6 + lVar8) >> (0x10U - *(int *)(lVar6 + 0x14) & 0x1f));
  }
  return 0;
}
```

## FUN_013d3268 header markers and dimensions only

```c
/* WARNING: Type propagation algorithm not settling */

void FUN_013d3268(long *param_1,short *param_2,ulong param_3)

{
  uint *puVar1;
  int iVar2;
  byte bVar3;
  byte bVar4;
  undefined1 uVar5;
  ushort uVar6;
  short sVar7;
  uint uVar8;
  uint uVar9;
  int iVar10;
  byte bVar11;
  uint uVar12;
  byte bVar13;
  ushort uVar14;
  bool bVar15;
  bool bVar16;
  int iVar17;
  ulong uVar18;
  ulong uVar19;
  uint uVar20;
  long lVar21;
  undefined4 uVar22;
  ulong uVar23;
  ulong uVar24;
  int iVar25;
  uint uVar26;
  long lVar27;
  ushort *puVar28;
  undefined6 local_470;
  ushort uStack_46a;
  int local_468;
  undefined2 uStack_464;
  undefined2 uStack_462;
  uint *puStack_460;
  uint local_458;
  long local_70;

  local_70 = *(long *)PTR____stack_chk_guard_07a703a0;
  if (param_3 < 2) goto LAB_013d32a0;
  if (*param_2 != 0x10ff) {
    uVar18 = 0xffff5eee;
    goto LAB_013d334c;
  }
  if (param_3 < 4) goto LAB_013d32a0;
  if (param_2[1] != 0x50ff) {
    uVar18 = 0xffff5edf;
    goto LAB_013d334c;
  }
  if (5 < param_3) {
    lVar27 = *param_1;
    uVar6 = param_2[2];
    uVar20 = (uint)(uVar6 >> 8) | (uVar6 & 0xff00ff) << 8;
    uVar18 = (ulong)uVar20;
    if (uVar20 < 3) {
      if (uVar6 != 0x200) {
        uVar18 = 0xffff5eed;
        goto LAB_013d334c;
      }
      puVar28 = (ushort *)0x0;
      bVar15 = false;
    }
    else {
      _bzero(&local_470,0x400);
      if (param_3 < uVar18 + 4) goto LAB_013d32a0;
      iVar17 = _memcmp(param_2 + 3,&local_470,uVar18 - 2);
      bVar15 = iVar17 != 0;
      if (iVar17 == 0) {
        puVar28 = (ushort *)0x0;
      }
      else {
        if (uVar18 - 2 < 0x10) {
          uVar18 = 0xffff5ede;
          goto LAB_013d334c;
        }
        if ((long)param_3 < 0x26) goto LAB_013d32a0;
        if (*(long *)(param_2 + 3) != 0x5f544341544e4f43 ||
            *(long *)(param_2 + 7) != 0x5f5849504f544e49) {
          uVar18 = 0xffff5edd;
          goto LAB_013d334c;
        }
        puVar28 = (ushort *)(param_2 + 0xb);
        if (((*puVar28 & 0x3f00) != 0) ||
           (*(long *)(param_2 + 0xc) != CONCAT26(uStack_46a,local_470) ||
            *(long *)(param_2 + 0xf) != CONCAT26(uStack_464,CONCAT42(local_468,uStack_46a)))) {
          uVar18 = 0xffff5edc;
          goto LAB_013d334c;
        }
      }
    }
    if (uVar18 + 6 <= param_3) {
      if (*(short *)((long)param_2 + uVar18 + 4) != 0x12ff) {
        uVar18 = 0xffff5edb;
        goto LAB_013d334c;
      }
      if ((param_3 < uVar18 + 8) || (uVar24 = uVar18 + 0x20, param_3 < uVar24)) goto LAB_013d32a0;
      puVar1 = (uint *)((long)param_2 + uVar18 + 8);
      if (((short)puVar1[3] != 0) ||
         ((((*(char *)((long)puVar1 + 0x11) != '\x04' || (*(char *)((long)puVar1 + 0x12) != '\b'))
           || ((puVar1[5] & 0xe8008e0f) != 4)) || (0x14 < *(byte *)((long)puVar1 + 0x13))))) {
        uVar18 = 0xffff5ed7;
        goto LAB_013d334c;
      }
      bVar3 = *(byte *)((long)param_2 + uVar18 + 7);
      uVar5 = *(undefined1 *)((long)param_2 + uVar18 + 6);
      *(uint *)(lVar27 + 0x18) = (uint)(byte)puVar1[4];
      uVar20 = (*puVar1 & 0xff00ff00) >> 8 | (*puVar1 & 0xff00ff) << 8;
      *(ulong *)(lVar27 + 0x88) = (ulong)(uVar20 >> 0x10 | uVar20 << 0x10);
      *(uint *)(lVar27 + 8) =
           (uint)(ushort)((ushort)puVar1[2] >> 8) | ((ushort)puVar1[2] & 0xff00ff) << 8;
      *(uint *)(lVar27 + 0xc) =
           (uint)(*(ushort *)((long)puVar1 + 10) >> 8) |
           (*(ushort *)((long)puVar1 + 10) & 0xff00ff) << 8;
      FUN_013d47dc(lVar27);
      if (bVar15) {
        FUN_013d4864(&local_470,(long)param_2 + uVar24,(int)param_3 - (int)uVar24);
        uVar14 = uStack_46a;
        uVar6 = *puVar28;
        if ((uVar6 >> 7 & 1) != 0) {
          uVar18 = CONCAT26(uStack_46a,local_470) << 2;
          local_470 = (undefined6)
```

## FUN_013b6938 task submission

```c
(uVar13 != 0) {
            uVar5 = uVar3 / uVar13;
          }
          plVar15[-1] = lVar24 + uVar4;
          *plVar15 = lVar24 + uVar1;
          plVar15[1] = lVar24 + uVar2;
          plVar15[2] = lVar24 + uVar5;
          plVar15 = plVar15 + 4;
          uVar14 = uVar14 - 4;
        } while (uVar14 != 0);
        if (uVar21 == uVar22) goto LAB_013b6de8;
      }
      lVar19 = uVar8 - uVar20;
      uVar16 = uVar23 * uVar20;
      plVar15 = local_b0 + uVar20;
      do {
        uVar16 = uVar23 + uVar16;
        uVar14 = 0;
        if (uVar13 != 0) {
          uVar14 = uVar16 / uVar13;
        }
        *plVar15 = lVar24 + uVar14;
        lVar19 = lVar19 + -1;
        plVar15 = plVar15 + 1;
      } while (lVar19 != 0);
    }
  }
LAB_013b6de8:
  uVar12 = FUN_013b667c(param_1,uVar9,param_3,FUN_013b5b44,0,local_b0,&local_64);
  if ((int)uVar12 == 0) {
    uVar12 = FUN_015033a4(*(undefined8 *)(param_1 + 0x28),local_64);
  }
  if (param_8 == 0) {
    FUN_0031bd7c(lVar10);
  }
joined_r0x013b6a0c:
  if (param_9 == 0) {
    FUN_0031bd7c(lVar11);
  }
  FUN_014d6144(*(undefined8 *)(*(long *)(param_1 + 0x28) + 0x28));
  return uVar12;
}
```

## FUN_013b667c callback submission

```c
iVar2 = FUN_01404820(param_3);
    if ((uVar9 & 0xf) != 0 && iVar2 == 0xe) {
      return 0xffff5fb7;
    }
    if ((uVar9 & 0xf) != 0 && (*(byte *)(lVar8 + 0x34) & 1) != 0) {
      return 0xffff5fbb;
    }
  }
  uVar3 = FUN_01404820(param_3);
  *puVar7 = uVar3;
  local_80 = param_1;
  uStack_78 = param_4;
  local_70 = param_5;
  uStack_68 = param_6;
  uVar4 = FUN_01502bb8(*(undefined8 *)(param_1 + 10),&local_80,param_7);
  return uVar4;
}
```

## FUN_013b5b44 output callback (address-taken)

```c
long FUN_013b5b44(undefined8 param_1,int param_2,uint param_3,ulong param_4,long param_5)

{
  long lVar1;

  if ((int)param_3 < *(int *)(param_5 + 0x40)) {
    lVar1 = *(long *)(param_5 + (long)param_2 * 8 + 0x20);
    if (param_4 <= (ulong)(lVar1 * 4)) {
      if (*(int *)(param_5 + 0x44) != 0) {
        param_3 = *(int *)(param_5 + 0x40) + ~param_3;
      }
      return *(long *)(param_5 + (long)param_2 * 8) + lVar1 * (int)param_3 * 4;
    }
  }
  else {
    FUN_01e78c34();
  }
                    /* WARNING: Subroutine does not return */
  FUN_01e78c0c();
}
```

