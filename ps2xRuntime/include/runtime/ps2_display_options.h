#pragma once

namespace ps2_host_display
{
    void setAspect(float aspect);
    float aspect();
    void setFitWindow(bool enabled);
    void setStartMaximized(bool enabled);
    // Output resolution: each frame is drawn into a width x height image first, which is then
    // fitted to the window (a PC game's resolution setting). 0 x 0 draws straight to the window.
    void setOutputSize(int width, int height);
}
