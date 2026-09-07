/**
 * @file mingw_compat.h
 * @brief ONNX Runtime MinGW兼容性头文件
 * @details ONNX Runtime的C API头文件(onnxruntime_c_api.h)在Windows平台下
 *          依赖MSVC的SAL注解(如_Frees_ptr_opt_、_In_、_Out_等)。
 *          MSVC通过<sal.h>提供这些注解，但MinGW没有此头文件。
 *          本文件在MinGW编译环境下定义这些SAL注解为空宏，解决编译错误。
 *          使用方法：在包含任何ONNX Runtime头文件之前包含本文件
 * @author 先瞳编码, 关注微信公众号"先瞳编码"，获取最新技术分享
 */

#ifndef ONNXRUNTIME_MINGW_COMPAT_H
#define ONNXRUNTIME_MINGW_COMPAT_H

// 仅在MinGW环境下需要此兼容层（MSVC自带sal.h）
#if defined(__MINGW32__) || defined(__MINGW64__)

// SAL2注解：在MinGW下定义为空宏，消除编译器的"未声明标识符"错误
// 这些注解仅用于代码静态分析，不影响运行时行为
#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif

#ifndef _In_
#define _In_
#endif

#ifndef _In_z_
#define _In_z_
#endif

#ifndef _In_opt_
#define _In_opt_
#endif

#ifndef _In_opt_z_
#define _In_opt_z_
#endif

#ifndef _Out_
#define _Out_
#endif

#ifndef _Outptr_
#define _Outptr_
#endif

#ifndef _Out_opt_
#define _Out_opt_
#endif

#ifndef _Inout_
#define _Inout_
#endif

#ifndef _Inout_opt_
#define _Inout_opt_
#endif

#ifndef _Ret_maybenull_
#define _Ret_maybenull_
#endif

#ifndef _Ret_notnull_
#define _Ret_notnull_
#endif

#ifndef _Outptr_result_maybenull_
#define _Outptr_result_maybenull_
#endif

#ifndef _In_reads_
#define _In_reads_(x)
#endif

#ifndef _Out_writes_
#define _Out_writes_(x)
#endif

#ifndef _In_reads_bytes_
#define _In_reads_bytes_(x)
#endif

#ifndef _Out_writes_bytes_
#define _Out_writes_bytes_(x)
#endif

#ifndef _Inout_updates_
#define _Inout_updates_(x)
#endif

#ifndef _Inout_updates_bytes_
#define _Inout_updates_bytes_(x)
#endif

#ifndef _Success_
#define _Success_(x)
#endif

#ifndef _Pre_opt_valid_
#define _Pre_opt_valid_
#endif

#ifndef _Post_invalid_
#define _Post_invalid_
#endif

#ifndef _Post_valid_
#define _Post_valid_
#endif

#ifndef _Deref_in_opt_
#define _Deref_in_opt_
#endif

#ifndef _Deref_inout_opt_
#define _Deref_inout_opt_
#endif

#ifndef _Deref_out_
#define _Deref_out_
#endif

#ifndef _Deref_out_opt_
#define _Deref_out_opt_
#endif

#ifndef _Deref_inout_
#define _Deref_inout_
#endif

#ifndef _Deref_inout_bytecount_
#define _Deref_inout_bytecount_(x)
#endif

#ifndef _Outptr_result_buffer_
#define _Outptr_result_buffer_(x)
#endif

#ifndef _Outptr_result_buffer_maybenull_
#define _Outptr_result_buffer_maybenull_(x)
#endif

#ifndef _Out_writes_bytes_to_
#define _Out_writes_bytes_to_(x, y)
#endif

#ifndef _In_opt_z_
#define _In_opt_z_
#endif

#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif

#ifndef _ACRTIMP
#define _ACRTIMP
#endif

#ifndef _VCRTIMP
#define _VCRTIMP
#endif

#endif // __MINGW32__ || __MINGW64__

#endif // ONNXRUNTIME_MINGW_COMPAT_H
