/*
* Vulkan Example - Font rendering using signed distance fields
*
* Font generated using https://github.com/libgdx/libgdx/wiki/Hiero
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanExampleBase.h"

// Vertex layout for this example
struct Vertex {
    float pos[3];
    float uv[2];
};

// AngelCode .fnt format structs and classes
struct bmchar {
    uint32_t x, y;
    uint32_t width;
    uint32_t height;
    int32_t xoffset;
    int32_t yoffset;
    int32_t xadvance;
    uint32_t page;
};

// Quick and dirty : complete ASCII table
// Only chars present in the .fnt are filled with data!
std::array<bmchar, 255> fontChars;

int32_t nextValuePair(std::stringstream* stream) {
    std::string pair;
    *stream >> pair;
    uint32_t spos = (uint32_t)pair.find("=");
    std::string value = pair.substr(spos + 1);
    int32_t val = std::stoi(value);
    return val;
}

class VulkanExample : public vkx::ExampleBase {
public:
    bool splitScreen = true;

    struct {
        vks::texture::Texture2D fontSDF;
        vks::texture::Texture2D fontBitmap;
    } textures;

    struct {
        vks::Buffer vertices;
        vks::Buffer indices;
        uint32_t count;
    } geometry;

    struct {
        vks::Buffer vs;
        vks::Buffer fs;
    } uniformData;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
    } uboVS;

    struct UboFS {
        glm::vec4 outlineColor = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
        float outlineWidth = 0.6f;
        float outline = true;
    } uboFS;

    struct {
        vk::Pipeline sdf;
        vk::Pipeline bitmap;
    } pipelines;

    struct {
        vk::DescriptorSet sdf;
        vk::DescriptorSet bitmap;
    } descriptorSets;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        camera.dolly(-2.0f);
        title = "Vulkan Example - Distance field fonts";
        camera.setPerspective(splitScreen ? 30.0f : 45.0f, (float)size.width / (float)(size.height * ((splitScreen) ? 0.5f : 1.0f)), 0.001f, 256.0f);
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        // Clean up texture resources
        textures.fontSDF.destroy();
        textures.fontBitmap.destroy();

        device.destroy(pipelines.bitmap);
        device.destroy(pipelines.sdf);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        geometry.vertices.destroy();
        geometry.indices.destroy();
        uniformData.vs.destroy();
        uniformData.fs.destroy();
    }

    struct membuf : std::streambuf {
        membuf(char const* base, size_t size) {
            char* p(const_cast<char*>(base));
            this->setg(p, p, p + size);
        }
    };
    struct imemstream : virtual membuf, std::istream {
        imemstream(char const* base, size_t size)
            : membuf(base, size)
            , std::istream(static_cast<std::streambuf*>(this)) {}
    };

    // Basic parser fpr AngelCode bitmap font format files
    // See http://www.angelcode.com/products/bmfont/doc/file_format.html for details
    void loadAssets() override {
        textures.fontSDF.loadFromFile(context, getAssetPath() + "textures/font_sdf_rgba.ktx", vk::Format::eR8G8B8A8Unorm);
        textures.fontBitmap.loadFromFile(context, getAssetPath() + "textures/font_bitmap_rgba.ktx", vk::Format::eR8G8B8A8Unorm);

        std::string fileName = getAssetPath() + "font.fnt";

        vks::file::withBinaryFileContents(fileName, [&](size_t size, const void* data) {
            imemstream istream((const char*)data, size);
            assert(istream.good());
            while (!istream.eof()) {
                std::string line;
                std::stringstream lineStream;
                std::getline(istream, line);
                lineStream << line;

                std::string info;
                lineStream >> info;

                if (info == "char") {
                    std::string pair;

                    // char id
                    uint32_t charid = nextValuePair(&lineStream);
                    // Char properties
                    fontChars[charid].x = nextValuePair(&lineStream);
                    fontChars[charid].y = nextValuePair(&lineStream);
                    fontChars[charid].width = nextValuePair(&lineStream);
                    fontChars[charid].height = nextValuePair(&lineStream);
                    fontChars[charid].xoffset = nextValuePair(&lineStream);
                    fontChars[charid].yoffset = nextValuePair(&lineStream);
                    fontChars[charid].xadvance = nextValuePair(&lineStream);
                    fontChars[charid].page = nextValuePair(&lineStream);
                }
            }
        });
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        vk::Viewport viewport = vks::util::viewport((float)size.width, (splitScreen) ? (float)size.height / 2.0f : (float)size.height, 0.0f, 1.0f);
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.setScissor(0, vks::util::rect2D(size));

        // Signed distance field font
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.sdf, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.sdf);
        cmdBuffer.bindVertexBuffers(0, geometry.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(geometry.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(geometry.count, 1, 0, 0, 0);

        // Linear filtered bitmap font
        if (splitScreen) {
            viewport.y += viewport.height;
            cmdBuffer.setViewport(0, viewport);
            cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.bitmap, nullptr);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.bitmap);
            cmdBuffer.bindVertexBuffers(0, geometry.vertices.buffer, { 0 });
            cmdBuffer.bindIndexBuffer(geometry.indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuffer.drawIndexed(geometry.count, 1, 0, 0, 0);
        }
    }

    // todo : function fill buffer with quads from font

    // Creates a vertex buffer containing quads for the passed text
    void generateText(std::string text) {
        std::vector<Vertex> vertexBuffer;
        std::vector<uint32_t> indexBuffer;
        uint32_t indexOffset = 0;

        float w = (float)textures.fontSDF.extent.width;

        float posx = 0.0f;
        float posy = 0.0f;

        for (uint32_t i = 0; i < text.size(); i++) {
            bmchar* charInfo = &fontChars[(int)text[i]];

            if (charInfo->width == 0)
                charInfo->width = 36;

            float charw = ((float)(charInfo->width) / 36.0f);
            float dimx = 1.0f * charw;
            float charh = ((float)(charInfo->height) / 36.0f);
            float dimy = 1.0f * charh;
            posy = 1.0f - charh;

            float us = charInfo->x / w;
            float ue = (charInfo->x + charInfo->width) / w;
            float ts = charInfo->y / w;
            float te = (charInfo->y + charInfo->height) / w;

            float xo = charInfo->xoffset / 36.0f;
            float yo = charInfo->yoffset / 36.0f;

            vertexBuffer.push_back({ { posx + dimx + xo, posy + dimy, 0.0f }, { ue, te } });
            vertexBuffer.push_back({ { posx + xo, posy + dimy, 0.0f }, { us, te } });
            vertexBuffer.push_back({ { posx + xo, posy, 0.0f }, { us, ts } });
            vertexBuffer.push_back({ { posx + dimx + xo, posy, 0.0f }, { ue, ts } });

            std::array<uint32_t, 6> indices = { 0, 1, 2, 2, 3, 0 };
            for (auto& index : indices) {
                indexBuffer.push_back(indexOffset + index);
            }
            indexOffset += 4;

            float advance = ((float)(charInfo->xadvance) / 36.0f);
            posx += advance;
        }
        geometry.count = (uint32_t)indexBuffer.size();

        // Center
        for (auto& v : vertexBuffer) {
            v.pos[0] -= posx / 2.0f;
            v.pos[1] -= 0.5f;
        }

        geometry.vertices = context.stageToDeviceBuffer<Vertex>(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);
        geometry.indices = context.stageToDeviceBuffer<uint32_t>(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    vks::model::VertexLayout vertexLayout{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_UV,
    } };

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{ vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 4),
                                                       vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2) };
        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader image sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };

        // Signed distance front descriptor set
        descriptorSets.sdf = device.allocateDescriptorSets(allocInfo)[0];

        // Default font rendering descriptor set
        descriptorSets.bitmap = device.allocateDescriptorSets(allocInfo)[0];

        // vk::Image descriptor for the color map texture
        vk::DescriptorImageInfo texSdfDescriptor{ textures.fontSDF.sampler, textures.fontSDF.view, vk::ImageLayout::eGeneral };
        vk::DescriptorImageInfo texBmpDescriptor{ textures.fontBitmap.sampler, textures.fontBitmap.view, vk::ImageLayout::eGeneral };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // SDF text version
            // Binding 0 : Vertex shader uniform buffer
            vk::WriteDescriptorSet{ descriptorSets.sdf, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vs.descriptor },
            // Binding 1 : Fragment shader texture sampler
            vk::WriteDescriptorSet{ descriptorSets.sdf, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texSdfDescriptor },
            // Binding 2 : Fragment shader uniform buffer
            vk::WriteDescriptorSet{ descriptorSets.sdf, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.fs.descriptor },

            // Bitmap text version

            // Binding 0 : Vertex shader uniform buffer
            vk::WriteDescriptorSet{ descriptorSets.bitmap, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vs.descriptor },
            // Binding 1 : Fragment shader texture sampler
            vk::WriteDescriptorSet{ descriptorSets.bitmap, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texBmpDescriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        using BF = vk::BlendFactor;
        using BO = vk::BlendOp;
        // SDF font rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout, renderPass };
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        builder.depthStencilState = false;
        builder.colorBlendState.blendAttachmentStates[0] = { VK_TRUE, BF::eOne, BF::eOneMinusSrcAlpha, BO::eAdd, BF::eOne, BF::eZero, BO::eAdd };
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        builder.loadShader(getAssetPath() + "shaders/distancefieldfonts/sdf.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/distancefieldfonts/sdf.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.sdf = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Bitmap font rendering pipeline
        builder.loadShader(getAssetPath() + "shaders/distancefieldfonts/bitmap.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/distancefieldfonts/bitmap.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.bitmap = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vs = context.createUniformBuffer(uboVS);

        // Fragment sahder uniform buffer block
        // Contains font rendering parameters
        uniformData.fs = context.createUniformBuffer(uboFS);

        updateUniformBuffers();
        updateFontSettings();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.model = camera.matrices.view;
        uniformData.vs.copy(uboVS);
    }

    void updateFontSettings() { uniformData.fs.copy(uboFS); }

    void prepare() override {
        ExampleBase::prepare();
        generateText("Vulkan");
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            bool outline = (uboFS.outline == 1.0f);
            if (ui.checkBox("Outline", &outline)) {
                uboFS.outline = outline ? 1.0f : 0.0f;
                updateFontSettings();
            }
            if (ui.checkBox("Splitscreen", &splitScreen)) {
                camera.setPerspective(splitScreen ? 30.0f : 45.0f, (float)size.width / (float)(size.height * ((splitScreen) ? 0.5f : 1.0f)), 0.001f, 256.0f);
                buildCommandBuffers();
                updateUniformBuffers();
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
