# C++ Vulkan examples and demos

<img src="./documentation/images/vulkanlogoscene.png" alt="Vulkan demo scene" height="256px"><img src="./documentation/images/c_0.jpg" alt="C++" height="256px">

This is a fork of [Sascha Willems](https://github.com/SaschaWillems) excellent [Vulkan examples](https://github.com/SaschaWillems/Vulkan) with some modifications.

* All of the code has been ported to use the [Vulkan C++ API](https://github.com/KhronosGroup/Vulkan-Hpp)
* Vulkan 1.3 is being targeted as the base level, in order to use Synchronization2, Timeline Semaphores, and other former extensions that have been folded into core
* Memory management has been migrated almost entirely to use the [Vulkan Memory Allocator](https://gpuopen.com/vulkan-memory-allocator/)
* External dependencies are being drawn from VCPKG rather than being included in the repository
* All platform specific code for Windows and Linux has been consolidated to use [GLFW 3.2](http://www.glfw.org/)
* Enable validation layers by default when building in debug mode
* Avoid excessive use of vkDeviceWaitIdle and vkQueueWaitIdle
* Shaders are now built at build time and made available to the application as headers containing constexpr arrays of uint32_t
* ~Avoid excessive use of explicit image layout transitions, instead using implicit transitions via the RenderPass and Subpass definitions~
  * This has been superceded with the transition to using the Dynamic Rendering feature in Vulkan 1.3, as implicit transitions are no longer possible

# Building

Use the provided CMakeLists.txt for use with [CMake](https://cmake.org) to generate a build configuration for your toolchain.  Using 64 bit builds is strongly recommended.

# Examples

This information comes from the [original repository readme](https://github.com/SaschaWillems/Vulkan/blob/master/README.md)

## [Beginner Examples](EXAMPLES_INIT.md)

## [Basic Technique Examples](EXAMPLES_BASIC.md)

## [Offscreen Rendering Examples](EXAMPLES_OFFSCREEN.md)

## [VR Examples](EXAMPLES_VR.md)

## [Compute Examples](EXAMPLES_COMPUTE.md)

## [Broken Examples](EXAMPLES_BROKEN.md)

# Credits & Thanks

Special thanks to...
 - [Sascha Willems](https://github.com/SaschaWillems) for his original work on these examples
 - [Baldur Karlsson](https://github.com/baldurk) for his work on RenderDoc, without which I'm certain I would still be staring at a black screen
 - [Andreas Süßenbach](https://github.com/asuessenbach) and [Markus Tavenrath](https://github.com/mtavenrath) for the development of the C++ Vulkan API
 - [LunarG](https://vulkan.lunarg.com) for their work on the Vulkan SDK and validation layers

Thanks to the authors of these libraries :
- [Vulkan Memory Allocator](https://gpuopen.com/vulkan-memory-allocator/)
- [OpenGL Mathematics (GLM)](https://github.com/g-truc/glm)
- [OpenGL Image (GLI)](https://github.com/g-truc/gli)
- [Open Asset Import Library](https://github.com/assimp/assimp)


## Attributions / Licenses
Please note that (some) models and textures use separate licenses. Please comply to these when redistributing or using them in your own projects :
- Cubemap used in cubemap example by [Emil Persson(aka Humus)](http://www.humus.name/)
- Armored knight model used in deferred example by [Gabriel Piacenti](http://opengameart.org/users/piacenti)
- Voyager model by [NASA](http://nasa3d.arc.nasa.gov/models)
- Astroboy COLLADA model copyright 2008 Sony Computer Entertainment Inc.
- Old deer model used in tessellation example by [Čestmír Dammer](http://opengameart.org/users/cdmir)
- Hidden treasure scene used in pipeline and debug marker examples by [Laurynas Jurgila](http://www.blendswap.com/user/PigArt)
- Textures used in some examples by [Hugues Muller](http://www.yughues-folio.com)
- Updated compute particle system shader by [Lukas Bergdoll](https://github.com/Voultapher)
- Vulkan scene model (and derived models) by [Dominic Agoro-Ombaka](http://www.agorodesign.com/) and [Sascha Willems](http://www.saschawillems.de)
- Vulkan and the Vulkan logo are trademarks of the [Khronos Group Inc.](http://www.khronos.org)

## External resources
- [LunarG Vulkan SDK](https://vulkan.lunarg.com)
- [Official list of Vulkan resources](https://www.khronos.org/vulkan/resources)
- [Vulkan API specifications](https://www.khronos.org/registry/vulkan/specs/1.0/apispec.html) ([quick reference cards](https://www.khronos.org/registry/vulkan/specs/1.0/refguide/Vulkan-1.0-web.pdf))
- [SPIR-V specifications](https://www.khronos.org/registry/spir-v/specs/1.0/SPIRV.html)
