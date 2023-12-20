/*
* Vulkan Example - OpenGL interoperability example
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>
#include <vks/texture.hpp>

// FIXME make work on non-Win32 platforms
#if defined(WIN32)
struct ShareHandles {
    HANDLE memory{ INVALID_HANDLE_VALUE };
    HANDLE glReady{ INVALID_HANDLE_VALUE };
    HANDLE glComplete{ INVALID_HANDLE_VALUE };
};

static const uint32_t SHARED_TEXTURE_DIMENSION = 512;

class TextureGenerator {
public:
    static const std::string VERTEX_SHADER;
    static const std::string FRAGMENT_SHADER;

    void init(ShareHandles& handles, uint64_t memorySize) {
        glfw::Window::init();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);

        // Window doesn't need to be large, it only exists to give us a GL context
        window.createWindow(uvec2{ 10 });
        window.makeCurrent();

        gl::init();
        gl::setupDebugLogging();

        window.showWindow(false);
        program = gl::buildProgram(VERTEX_SHADER, FRAGMENT_SHADER);
        startTime = glfwGetTime();

        glDisable(GL_DEPTH_TEST);

        // Create the texture for the FBO color attachment.
        // This only reserves the ID, it doesn't allocate memory
        glCreateTextures(GL_TEXTURE_2D, 1, &color);

        // Import semaphores
        glGenSemaphoresEXT(1, &glReady);
        glGenSemaphoresEXT(1, &glComplete);

        // Platform specific import.  On non-Win32 systems use glImportSemaphoreFdEXT instead
        glImportSemaphoreWin32HandleEXT(glReady, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handles.glReady);
        glImportSemaphoreWin32HandleEXT(glComplete, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handles.glComplete);

        // Import memory
        glCreateMemoryObjectsEXT(1, &mem);
        // Platform specific import.  On non-Win32 systems use glImportMemoryFdEXT instead
        glImportMemoryWin32HandleEXT(mem, memorySize, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handles.memory);

        // Use the imported memory as backing for the OpenGL texture.  The internalFormat, dimensions
        // and mip count should match the ones used by Vulkan to create the image and determine it's memory
        // allocation.
        glTextureStorageMem2DEXT(color, 1, GL_RGBA8, SHARED_TEXTURE_DIMENSION, SHARED_TEXTURE_DIMENSION, mem, 0);

        // The remaining initialization code is all standard OpenGL
        glCreateFramebuffers(1, &fbo);
        glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, color, 0);
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glUseProgram(program);
        glProgramUniform3f(program, 0, (float)SHARED_TEXTURE_DIMENSION, (float)SHARED_TEXTURE_DIMENSION, 0.0f);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
        glViewport(0, 0, SHARED_TEXTURE_DIMENSION, SHARED_TEXTURE_DIMENSION);
    }

    void destroy() {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBindVertexArray(0);
        glUseProgram(0);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &color);
        glDeleteSemaphoresEXT(1, &glReady);
        glDeleteSemaphoresEXT(1, &glComplete);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(program);
        glFlush();
        glFinish();
        window.destroyWindow();
    }

    void render() {
        // The GL shader animates the image, so provide the time as input
        glProgramUniform1f(program, 1, (float)(glfwGetTime() - startTime));

        // Wait (on the GPU side) for the Vulkan semaphore to be signaled
        GLenum srcLayout = GL_LAYOUT_COLOR_ATTACHMENT_EXT;
        glWaitSemaphoreEXT(glReady, 0, nullptr, 1, &color, &srcLayout);

        // Draw to the framebuffer
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // Once drawing is complete, signal the Vulkan semaphore indicating
        // it can continue with it's render
        GLenum dstLayout = GL_LAYOUT_SHADER_READ_ONLY_EXT;
        glSignalSemaphoreEXT(glComplete, 0, nullptr, 1, &color, &dstLayout);

        // When using synchronization across multiple GL context, or in this case
        // across OpenGL and another API, it's critical that an operation on a
        // synchronization object that will be waited on in another context or API
        // is flushed to the GL server.
        //
        // Failure to flush the operation can cause the GL driver to sit and wait for
        // sufficient additional commands in the buffer before it flushes automatically
        // but depending on how the waits and signals are structured, this may never
        // occur.
        glFlush();
    }

private:
    GLuint glReady{ 0 }, glComplete{ 0 };
    GLuint color = 0;
    GLuint fbo = 0;
    GLuint vao = 0;
    GLuint program = 0;
    GLuint mem = 0;
    double startTime;
    glfw::Window window;
};

const std::string TextureGenerator::VERTEX_SHADER = R"SHADER(
#version 450 core

const vec4 VERTICES[] = vec4[](
    vec4(-1.0, -1.0, 0.0, 1.0), 
    vec4( 1.0, -1.0, 0.0, 1.0),    
    vec4(-1.0,  1.0, 0.0, 1.0),
    vec4( 1.0,  1.0, 0.0, 1.0)
);   

void main() { gl_Position = VERTICES[gl_VertexID]; }

)SHADER";

const std::string TextureGenerator::FRAGMENT_SHADER = R"SHADER(
#version 450 core

const vec4 iMouse = vec4(0.0); 

layout(location = 0) out vec4 outColor;

layout(location = 0) uniform vec3 iResolution;
layout(location = 1) uniform float iTime;

vec3 hash3( vec2 p )
{
    vec3 q = vec3( dot(p,vec2(127.1,311.7)), 
                   dot(p,vec2(269.5,183.3)), 
                   dot(p,vec2(419.2,371.9)) );
    return fract(sin(q)*43758.5453);
}

float iqnoise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
        
    float k = 1.0+63.0*pow(1.0-v,4.0);
    
    float va = 0.0;
    float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = vec2( float(i),float(j) );
        vec3 o = hash3( p + g )*vec3(u,u,1.0);
        vec2 r = g - f + o.xy;
        float d = dot(r,r);
        float ww = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
        va += o.z*ww;
        wt += ww;
    }
    
    return va/wt;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord.xy / iResolution.xx;

    vec2 p = 0.5 - 0.5*sin( iTime*vec2(1.01,1.71) );
    
    if( iMouse.w>0.001 ) p = vec2(0.0,1.0) + vec2(1.0,-1.0)*iMouse.xy/iResolution.xy;
    
    p = p*p*(3.0-2.0*p);
    p = p*p*(3.0-2.0*p);
    p = p*p*(3.0-2.0*p);
    
    float f = iqnoise( 24.0*uv, p.x, p.y );
    
    fragColor = vec4( f, f, f, 1.0 );
}

void main() { mainImage(outColor, gl_FragCoord.xy); }

)SHADER";

// Vertex layout for this example
struct Vertex {
    float pos[3];
    float uv[2];
    float normal[3];
};

// The bulk of this example is the same as the existing texture example.
// However, instead of loading a texture from a file, it relies on an OpenGL
// shader to populate the texture.
class OpenGLInteropExample : public vkx::ExampleBase {
    using Parent = ExampleBase;

public:
    struct SharedResources {
        vks::Image texture;
        struct {
            vk::Semaphore glReady;
            vk::Semaphore glComplete;
        } semaphores;
        vk::CommandBuffer transitionCmdBuf;
        ShareHandles handles;
        vk::Device device;

        void init(const vks::Context& context) {
            device = context.device;
            vk::DispatchLoaderDynamic dynamicLoader{ context.instance, &vkGetInstanceProcAddr, device, &vkGetDeviceProcAddr };
            {
                auto handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
                {
                    vk::SemaphoreCreateInfo sci;
                    vk::ExportSemaphoreCreateInfo esci;
                    sci.pNext = &esci;
                    esci.handleTypes = handleType;
                    semaphores.glReady = device.createSemaphore(sci);
                    semaphores.glComplete = device.createSemaphore(sci);
                }
                handles.glReady = device.getSemaphoreWin32HandleKHR({ semaphores.glReady, handleType }, dynamicLoader);
                handles.glComplete = device.getSemaphoreWin32HandleKHR({ semaphores.glComplete, handleType }, dynamicLoader);
            }

            {
                vk::ImageCreateInfo imageCreateInfo;
                imageCreateInfo.imageType = vk::ImageType::e2D;
                imageCreateInfo.format = vk::Format::eR8G8B8A8Unorm;
                imageCreateInfo.mipLevels = 1;
                imageCreateInfo.arrayLayers = 1;
                imageCreateInfo.extent.depth = 1;
                imageCreateInfo.extent.width = SHARED_TEXTURE_DIMENSION;
                imageCreateInfo.extent.height = SHARED_TEXTURE_DIMENSION;
                imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
                texture.image = device.createImage(imageCreateInfo);
                texture.device = device;
                texture.format = imageCreateInfo.format;
                texture.extent = imageCreateInfo.extent;
            }

            {
                vk::MemoryRequirements memReqs = device.getImageMemoryRequirements(texture.image);
                vk::MemoryAllocateInfo memAllocInfo;
                vk::ExportMemoryAllocateInfo exportAllocInfo{ vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 };
                memAllocInfo.pNext = &exportAllocInfo;
                memAllocInfo.allocationSize = texture.allocSize = memReqs.size;
                memAllocInfo.memoryTypeIndex = context.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
                texture.memory = device.allocateMemory(memAllocInfo);
                device.bindImageMemory(texture.image, texture.memory, 0);
                handles.memory = device.getMemoryWin32HandleKHR({ texture.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 }, dynamicLoader);
            }

            {
                // Create sampler
                vk::SamplerCreateInfo samplerCreateInfo;
                samplerCreateInfo.magFilter = vk::Filter::eLinear;
                samplerCreateInfo.minFilter = vk::Filter::eLinear;
                samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
                // Max level-of-detail should match mip level count
                samplerCreateInfo.maxLod = (float)1;
                // Only enable anisotropic filtering if enabled on the devicec
                samplerCreateInfo.maxAnisotropy = context.deviceFeatures.samplerAnisotropy ? context.deviceProperties.limits.maxSamplerAnisotropy : 1.0f;
                samplerCreateInfo.anisotropyEnable = context.deviceFeatures.samplerAnisotropy;
                samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
                texture.sampler = device.createSampler(samplerCreateInfo);
            }

            {
                // Create image view
                vk::ImageViewCreateInfo viewCreateInfo;
                viewCreateInfo.viewType = vk::ImageViewType::e2D;
                viewCreateInfo.image = texture.image;
                viewCreateInfo.format = texture.format;
                viewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
                texture.view = context.device.createImageView(viewCreateInfo);
            }

            // Setup the command buffers used to transition the image between GL and VK
            transitionCmdBuf = context.createCommandBuffer();
            transitionCmdBuf.begin(vk::CommandBufferBeginInfo{});
            context.setImageLayout(transitionCmdBuf, texture.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                                   vk::ImageLayout::eColorAttachmentOptimal);
            transitionCmdBuf.end();
        }

        void destroy() {
            texture.destroy();
            device.destroy(semaphores.glComplete);
            device.destroy(semaphores.glReady);
        }

        void transitionToGl(const vk::Queue& queue, const vk::Semaphore& waitSemaphore) const {
            vk::SubmitInfo submitInfo;
            vk::PipelineStageFlags stageFlags = vk::PipelineStageFlagBits::eBottomOfPipe;
            submitInfo.pWaitDstStageMask = &stageFlags;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &waitSemaphore;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &semaphores.glReady;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &transitionCmdBuf;
            queue.submit({ submitInfo }, {});
        }
    } shared;

    TextureGenerator texGenerator;

    struct Geometry {
        uint32_t count{ 0 };
        vks::Buffer indices;
        vks::Buffer vertices;
    } geometry;

    vks::Buffer uniformDataVS;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 viewPos;
        float lodBias = 0.0f;
    } uboVS;

    struct {
        vk::Pipeline solid;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    OpenGLInteropExample() {
        camera.setRotation({ 0.0f, 15.0f, 0.0f });
        camera.dolly(-2.5f);
        title = "Vulkan Example - Texturing";

        context.requireExtensions({
            VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,    //
            VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME  //
        });

        context.requireDeviceExtensions({
            VK_KHR_MAINTENANCE1_EXTENSION_NAME,            //
                VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,     //
                VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,  //
#if defined(WIN32)
                VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,    //
                VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME  //
#else
                VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,    //
                VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME  //
#endif
        });
    }

    ~OpenGLInteropExample() {
        shared.destroy();

        device.destroyPipeline(pipelines.solid);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        geometry.vertices.destroy();
        geometry.indices.destroy();

        device.destroyBuffer(uniformDataVS.buffer);
        device.freeMemory(uniformDataVS.memory);
    }

    void buildExportableImage() {
        shared.init(context);
        texGenerator.init(shared.handles, shared.texture.allocSize);
    }

    void updateCommandBufferPreDraw(const vk::CommandBuffer& commandBuffer) override {
        context.setImageLayout(commandBuffer, shared.texture.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
        vk::DeviceSize offsets = 0;
        cmdBuffer.bindVertexBuffers(0, geometry.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(geometry.indices.buffer, 0, vk::IndexType::eUint32);

        cmdBuffer.drawIndexed(geometry.count, 1, 0, 0, 0);
    }

    void generateQuad() {
        // Setup vertices for a single uv-mapped quad
#define DIM 1.0f
#define NORMAL { 0.0f, 0.0f, 1.0f }
        std::vector<Vertex> vertexBuffer = { { { DIM, DIM, 0.0f }, { 1.0f, 1.0f }, NORMAL },
                                             { { -DIM, DIM, 0.0f }, { 0.0f, 1.0f }, NORMAL },
                                             { { -DIM, -DIM, 0.0f }, { 0.0f, 0.0f }, NORMAL },
                                             { { DIM, -DIM, 0.0f }, { 1.0f, 0.0f }, NORMAL } };
#undef DIM
#undef NORMAL
        geometry.vertices = context.stageToDeviceBuffer<Vertex>(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        geometry.count = (uint32_t)indexBuffer.size();
        geometry.indices = context.stageToDeviceBuffer<uint32_t>(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 1 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 1 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader image sampler
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        // vk::Image descriptor for the color map texture
        vk::DescriptorImageInfo texDescriptor{ shared.texture.sampler, shared.texture.view, vk::ImageLayout::eShaderReadOnlyOptimal };
        device.updateDescriptorSets(
            {
                // Binding 0 : Vertex shader uniform buffer
                vk::WriteDescriptorSet{ descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataVS.descriptor },
                // Binding 1 : Fragment shader texture sampler
                vk::WriteDescriptorSet{ descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
            },
            {});
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.vertexInputState.bindingDescriptions = { { 0, sizeof(Vertex), vk::VertexInputRate::eVertex } };
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            { 0, 0, vk::Format::eR32G32B32Sfloat, 0 },
            { 1, 0, vk::Format::eR32G32Sfloat, sizeof(float) * 3 },
            { 2, 0, vk::Format::eR32G32B32Sfloat, sizeof(float) * 5 },
        };
        pipelineBuilder.loadShader(getAssetPath() + "shaders/texture/texture.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/texture/texture.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.solid = pipelineBuilder.create(context.pipelineCache);
    }

    void prepareUniformBuffers() {
        uniformDataVS = context.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, camera.position.z));
        uboVS.model = viewMatrix * glm::translate(glm::mat4(), glm::vec3(camera.position.x, camera.position.y, 0));
        uboVS.model = uboVS.model * glm::inverse(camera.matrices.skyboxView);
        uboVS.viewPos = glm::vec4(0.0f, 0.0f, -camera.position.z, 0.0f);
        uniformDataVS.copy(uboVS);
    }

    void prepare() override {
        Parent::prepare();
        generateQuad();
        prepareUniformBuffers();
        buildExportableImage();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffers(); }

    void draw() override {
        prepareFrame();
        shared.transitionToGl(queue, semaphores.acquireComplete);
        texGenerator.render();
        renderWaitSemaphores = { shared.semaphores.glComplete };
        drawCurrentCommandBuffer();
        submitFrame();
    }
};
#else
class OpenGLInteropExample {
public:
    void run() {}
};
#endif

RUN_EXAMPLE(OpenGLInteropExample)
