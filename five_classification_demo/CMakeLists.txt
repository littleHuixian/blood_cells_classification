cmake_minimum_required(VERSION 3.19)
project(five_classification_demo LANGUAGES CXX)

# 模型文件放在工程 model 目录，构建/安装时复制到程序目录
set(BLOOD_MODEL_ONNX
    "${CMAKE_CURRENT_SOURCE_DIR}/model/best_blood_cell_model_ir9.onnx")
set(BLOOD_MODEL_DATA
    "${CMAKE_CURRENT_SOURCE_DIR}/model/best_blood_cell_model_ir9.onnx.data")

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets)

if(APPLE)
    # macOS：使用项目内置的官方 arm64 ONNX Runtime
    set(ONNXRUNTIME_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/macos"
        CACHE PATH "ONNX Runtime installation root" FORCE)
    set(OpenCV_DIR
        "/opt/opencv/lib/cmake/opencv5"
        CACHE PATH "Path containing OpenCVConfig.cmake" FORCE)
elseif(WIN32)
    # Windows：继续使用原工程的本机 ONNX Runtime / OpenCV
    set(ONNXRUNTIME_DIR
        "D:/onnxruntime-win-x64-1.16.3"
        CACHE PATH "ONNX Runtime installation root")
    set(OpenCV_DIR
        "D:/OpenCV453/build/install/x64/mingw/lib"
        CACHE PATH "Path containing OpenCVConfig.cmake")
else()
    message(FATAL_ERROR "five_classification_demo only supports macOS and Windows")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ONNX Runtime 动态库：macOS 直接链接项目内 dylib，
# Windows 运行用 D 盘 DLL，链接用本工程内生成的 MinGW 导入库
add_library(onnxruntime SHARED IMPORTED GLOBAL)
if(APPLE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
    )
elseif(WIN32)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/lib/libonnxruntime.a"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/include"
    )
endif()

qt_standard_project_setup()

# 让 images.qrc 真正被 rcc 编译进可执行文件
set(CMAKE_AUTORCC ON)

qt_add_executable(five_classification_demo
    WIN32 MACOSX_BUNDLE
    main.cpp
    bloodcell_classifier.cpp
    bloodcell_classifier.h
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    images.qrc
)

target_include_directories(five_classification_demo
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

if(APPLE)
    set_target_properties(five_classification_demo PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
        INSTALL_RPATH "@loader_path/../Frameworks"
    )
endif()

target_link_libraries(five_classification_demo
    PRIVATE
        Qt::Core
        Qt::Widgets
        onnxruntime
        ${OpenCV_LIBS}
)

# 将运行所需的 ONNX Runtime / OpenCV DLL 复制到可执行文件目录
if(WIN32)
    set(OPENCV_RUNTIME_DIR "D:/OpenCV453/build/install/x64/mingw/bin")
    add_custom_command(TARGET five_classification_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:five_classification_demo>/model"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${BLOOD_MODEL_ONNX}"
            "$<TARGET_FILE_DIR:five_classification_demo>/model/best_blood_cell_model_ir9.onnx"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${BLOOD_MODEL_DATA}"
            "$<TARGET_FILE_DIR:five_classification_demo>/model/best_blood_cell_model_ir9.onnx.data"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:five_classification_demo>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_RUNTIME_DIR}/libopencv_core453.dll"
            "$<TARGET_FILE_DIR:five_classification_demo>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_RUNTIME_DIR}/libopencv_imgproc453.dll"
            "$<TARGET_FILE_DIR:five_classification_demo>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_RUNTIME_DIR}/libopencv_imgcodecs453.dll"
            "$<TARGET_FILE_DIR:five_classification_demo>"
        VERBATIM
    )
    install(FILES
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        "${OPENCV_RUNTIME_DIR}/libopencv_core453.dll"
        "${OPENCV_RUNTIME_DIR}/libopencv_imgproc453.dll"
        "${OPENCV_RUNTIME_DIR}/libopencv_imgcodecs453.dll"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(TARGETS five_classification_demo
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET five_classification_demo
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})

install(FILES ${BLOOD_MODEL_ONNX} ${BLOOD_MODEL_DATA}
    DESTINATION ${CMAKE_INSTALL_BINDIR}/model
)
