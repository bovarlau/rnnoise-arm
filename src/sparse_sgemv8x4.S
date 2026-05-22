    .arch armv7-a
    .fpu neon

    .global sparse_sgemv8x4
    .type sparse_sgemv8x4, %function

sparse_sgemv8x4:
    // 函数参数：
    // r0 = out (float*)
    // r1 = w (const float*)
    // r2 = idx (const int*)
    // r3 = rows (int)
    // r4 = x (const float*)

    push {r4-r11, lr}         // 保存寄存器

    // 外循环：遍历每一行 (rows)
    mov r5, r3                // r5 = rows
    cmp r5, #0                // 检查 rows 是否为 0
    beq end_func              // 如果 rows 为 0，则结束

loop_rows:
    ldr r6, [r2], #4          // 加载当前行的列数 (cols)
    cmp r6, #0                // 如果列数为 0，跳过
    beq next_row

    // 内循环：处理每一列
    ldr r7, [r2], #4          // 读取列索引 pos
    add r8, r4, r7, lsl #2    // r8 = &x[pos]，x向量的地址

    vld1.32 {d8, d9}, [r8]    // 加载 x[pos] 的四个 float 元素 (xj0, xj1, xj2, xj3)

    // 加载 y 向量
    vld1.32 {d0, d1}, [r0]    // 加载 y[0] 到 y[3]
    add r9, r0, #16           // 计算 y[4] 到 y[7] 的地址
    vld1.32 {d2, d3}, [r9]    // 加载 y[4] 到 y[7]

    // 处理 xj0
    vld1.32 {d4, d5}, [r1]    // 加载 w[0] 到 w[7]
    vmla.f32 q0, q2, d8[0]    // y0 += w0 * xj0
    vmla.f32 q1, q3, d8[0]    // y1 += w1 * xj0

    // 处理 xj1
    add r10, r1, #32          // 计算 w[8] 到 w[15] 的地址
    vld1.32 {d6, d7}, [r10]   // 加载 w[8] 到 w[15]
    vmla.f32 q0, q3, d8[1]    // y0 += w8 * xj1
    vmla.f32 q1, q3, d8[1]    // y1 += w9 * xj1

    // 处理 xj2
    add r10, r1, #64          // 计算 w[16] 到 w[23] 的地址
    vld1.32 {d4, d5}, [r10]   // 加载 w[16] 到 w[23]
    vmla.f32 q0, q2, d9[0]    // y0 += w16 * xj2
    vmla.f32 q1, q3, d9[0]    // y1 += w17 * xj2

    // 处理 xj3
    add r10, r1, #96          // 计算 w[24] 到 w[31] 的地址
    vld1.32 {d6, d7}, [r10]   // 加载 w[24] 到 w[31]
    vmla.f32 q0, q2, d9[1]    // y0 += w24 * xj3
    vmla.f32 q1, q3, d9[1]    // y1 += w25 * xj3

    // 存回 y 向量
    vst1.32 {d0, d1}, [r0]    // 存回 y[0] 到 y[3]
    vst1.32 {d2, d3}, [r9]    // 存回 y[4] 到 y[7]

    add r1, r1, #128          // 权重指针前移 32 个 float（128 字节）

    // 跳回到外循环
    b loop_rows

next_row:
    subs r5, r5, #8           // 减少处理的行数
    cmp r5, #0                // 检查是否处理完所有行
    bne loop_rows             // 如果没有处理完，继续

end_func:
    pop {r4-r11, pc}          // 恢复寄存器并返回
