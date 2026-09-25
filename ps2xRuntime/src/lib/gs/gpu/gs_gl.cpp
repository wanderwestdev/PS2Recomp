#include "gs_gl.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace
{
    constexpr int kContextMajorVersion = 0x2091;
    constexpr int kContextMinorVersion = 0x2092;
    constexpr int kContextFlags = 0x2094;
    constexpr int kContextProfileMask = 0x9126;
    constexpr int kContextCoreProfileBit = 0x0001;

    using CreateContextAttribs = HGLRC(WINAPI *)(HDC, HGLRC, const int *);

    HMODULE openGlModule()
    {
        static HMODULE module = LoadLibraryA("opengl32.dll");
        return module;
    }

    void *loadProc(const char *name)
    {
        void *proc = reinterpret_cast<void *>(wglGetProcAddress(name));
        const auto value = reinterpret_cast<intptr_t>(proc);
        if (value == 0 || value == 1 || value == 2 || value == 3 || value == -1)
            proc = reinterpret_cast<void *>(GetProcAddress(openGlModule(), name));
        return proc;
    }

    template <typename T>
    bool load(T &target, const char *name, std::string &missing)
    {
        target = reinterpret_cast<T>(loadProc(name));
        if (!target)
        {
            missing += missing.empty() ? name : std::string(", ") + name;
            return false;
        }
        return true;
    }

    LRESULT CALLBACK windowProc(HWND window, UINT message, WPARAM wparam, LPARAM lparam)
    {
        return DefWindowProcA(window, message, wparam, lparam);
    }
}

namespace GSGL
{
    Context::~Context()
    {
        Destroy();
    }

    bool Context::Create(std::string &error)
    {
        Destroy();
        WNDCLASSA windowClass{};
        windowClass.style = CS_OWNDC;
        windowClass.lpfnWndProc = windowProc;
        windowClass.hInstance = GetModuleHandleA(nullptr);
        windowClass.lpszClassName = "PS2XGSGpuContext";
        RegisterClassA(&windowClass);
        HWND window = CreateWindowExA(0, windowClass.lpszClassName, "gs", WS_OVERLAPPEDWINDOW, 0, 0, 16, 16, nullptr, nullptr,
                                      windowClass.hInstance, nullptr);
        if (!window)
        {
            error = "CreateWindowEx failed";
            return false;
        }
        HDC dc = GetDC(window);
        PIXELFORMATDESCRIPTOR descriptor{};
        descriptor.nSize = sizeof(descriptor);
        descriptor.nVersion = 1;
        descriptor.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        descriptor.iPixelType = PFD_TYPE_RGBA;
        descriptor.cColorBits = 32;
        const int format = ChoosePixelFormat(dc, &descriptor);
        if (format == 0 || !SetPixelFormat(dc, format, &descriptor))
        {
            ReleaseDC(window, dc);
            DestroyWindow(window);
            error = "no pixel format";
            return false;
        }
        HGLRC legacy = wglCreateContext(dc);
        if (!legacy || !wglMakeCurrent(dc, legacy))
        {
            if (legacy)
                wglDeleteContext(legacy);
            ReleaseDC(window, dc);
            DestroyWindow(window);
            error = "wglCreateContext failed";
            return false;
        }
        const auto createAttribs = reinterpret_cast<CreateContextAttribs>(wglGetProcAddress("wglCreateContextAttribsARB"));
        HGLRC context = nullptr;
        if (createAttribs)
        {
            const int attributes[] = {kContextMajorVersion, 4, kContextMinorVersion, 6, kContextProfileMask, kContextCoreProfileBit,
                                      kContextFlags, 0, 0};
            context = createAttribs(dc, nullptr, attributes);
        }
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(legacy);
        if (!context || !wglMakeCurrent(dc, context))
        {
            if (context)
                wglDeleteContext(context);
            ReleaseDC(window, dc);
            DestroyWindow(window);
            error = "OpenGL 4.6 core context unavailable";
            return false;
        }
        m_window = window;
        m_dc = dc;
        m_context = context;

        std::string missing;
        Api &a = m_api;
        load(a.GenBuffers, "glGenBuffers", missing);
        load(a.DeleteBuffers, "glDeleteBuffers", missing);
        load(a.BindBuffer, "glBindBuffer", missing);
        load(a.BufferData, "glBufferData", missing);
        load(a.BufferSubData, "glBufferSubData", missing);
        load(a.GetBufferSubData, "glGetBufferSubData", missing);
        load(a.BindBufferBase, "glBindBufferBase", missing);
        load(a.BindBufferRange, "glBindBufferRange", missing);
        load(a.ClearBufferData, "glClearBufferData", missing);
        load(a.CopyBufferSubData, "glCopyBufferSubData", missing);
        load(a.BufferStorage, "glBufferStorage", missing);
        load(a.MapBufferRange, "glMapBufferRange", missing);
        load(a.UnmapBuffer, "glUnmapBuffer", missing);
        load(a.GenQueries, "glGenQueries", missing);
        load(a.DeleteQueries, "glDeleteQueries", missing);
        load(a.BeginQuery, "glBeginQuery", missing);
        load(a.EndQuery, "glEndQuery", missing);
        load(a.GetQueryObjectui64v, "glGetQueryObjectui64v", missing);
        load(a.CreateShader, "glCreateShader", missing);
        load(a.ShaderSource, "glShaderSource", missing);
        load(a.CompileShader, "glCompileShader", missing);
        load(a.GetShaderiv, "glGetShaderiv", missing);
        load(a.GetShaderInfoLog, "glGetShaderInfoLog", missing);
        load(a.DeleteShader, "glDeleteShader", missing);
        load(a.CreateProgram, "glCreateProgram", missing);
        load(a.AttachShader, "glAttachShader", missing);
        load(a.LinkProgram, "glLinkProgram", missing);
        load(a.GetProgramiv, "glGetProgramiv", missing);
        load(a.GetProgramInfoLog, "glGetProgramInfoLog", missing);
        load(a.DeleteProgram, "glDeleteProgram", missing);
        load(a.UseProgram, "glUseProgram", missing);
        load(a.GetUniformLocation, "glGetUniformLocation", missing);
        load(a.Uniform1ui, "glUniform1ui", missing);
        load(a.Uniform4ui, "glUniform4ui", missing);
        load(a.Uniform1uiv, "glUniform1uiv", missing);
        load(a.DispatchCompute, "glDispatchCompute", missing);
        load(a.MemBarrier, "glMemoryBarrier", missing);
        load(a.FenceSync, "glFenceSync", missing);
        load(a.ClientWaitSync, "glClientWaitSync", missing);
        load(a.DeleteSync, "glDeleteSync", missing);
        load(a.Finish, "glFinish", missing);
        load(a.Flush, "glFlush", missing);
        load(a.GetError, "glGetError", missing);
        load(a.GetString, "glGetString", missing);
        if (!missing.empty())
        {
            error = "missing GL functions: " + missing;
            Destroy();
            return false;
        }
        return true;
    }

    void Context::Destroy()
    {
        if (m_context)
        {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(static_cast<HGLRC>(m_context));
            m_context = nullptr;
        }
        if (m_dc && m_window)
            ReleaseDC(static_cast<HWND>(m_window), static_cast<HDC>(m_dc));
        m_dc = nullptr;
        if (m_window)
            DestroyWindow(static_cast<HWND>(m_window));
        m_window = nullptr;
        m_api = {};
    }
}
#else
namespace GSGL
{
    Context::~Context() = default;

    bool Context::Create(std::string &error)
    {
        error = "GPU GS backend is only implemented for Windows";
        return false;
    }

    void Context::Destroy()
    {
    }
}
#endif
