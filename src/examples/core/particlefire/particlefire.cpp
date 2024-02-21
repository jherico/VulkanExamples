/*
 * Vulkan Example - CPU based fire particle system
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>

#include <shaders/particlefire/normalmap.frag.inl>
#include <shaders/particlefire/normalmap.vert.inl>
#include <shaders/particlefire/particle.frag.inl>
#include <shaders/particlefire/particle.vert.inl>

#define PARTICLE_COUNT 512
#define PARTICLE_SIZE 10.0f

#define FLAME_RADIUS 8.0f

#define PARTICLE_TYPE_FLAME 0
#define PARTICLE_TYPE_SMOKE 1

struct Particle {
    glm::vec3 pos;
    glm::vec3 color;
    float alpha;
    float size;
    float rotation;
    uint32_t type;
    // Attributes not used in shader
    glm::vec3 vel;
    float rotationSpeed;
};

// Vertex layout for this example
class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        struct {
            vks::texture::Texture2D smoke;
            vks::texture::Texture2D fire;
            // We use a custom sampler to change some sampler
            // attributes required for rotation the uv coordinates
            // inside the shader for alpha blended textures
            vk::Sampler sampler;
        } particles;
        struct {
            vks::texture::Texture2D colorMap;
            vks::texture::Texture2D normalMap;
        } floor;
    } textures;

    struct {
        vks::model::Model environment;
        vk::DescriptorSet descriptorSet;
        vks::model::VertexLayout vertexLayout{ {
            vks::model::VERTEX_COMPONENT_POSITION,
            vks::model::VERTEX_COMPONENT_UV,
            vks::model::VERTEX_COMPONENT_NORMAL,
            vks::model::VERTEX_COMPONENT_TANGENT,
            vks::model::VERTEX_COMPONENT_BITANGENT,
        } };
    } meshes;

    glm::vec3 emitterPos = glm::vec3(0.0f, -FLAME_RADIUS + 2.0f, 0.0f);
    glm::vec3 minVel = glm::vec3(-3.0f, 0.5f, -3.0f);
    glm::vec3 maxVel = glm::vec3(3.0f, 7.0f, 3.0f);

    struct {
        vks::Buffer buffer;
        vks::model::VertexLayout vertexLayout{ {
            vks::model::VERTEX_COMPONENT_POSITION,
            vks::model::VERTEX_COMPONENT_COLOR,
            vks::model::VERTEX_COMPONENT_DUMMY_FLOAT,  // alpha
            vks::model::VERTEX_COMPONENT_DUMMY_FLOAT,  // size
            vks::model::VERTEX_COMPONENT_DUMMY_FLOAT,  // rotaton
            vks::model::VERTEX_COMPONENT_DUMMY_INT,    // type
            vks::model::VERTEX_COMPONENT_DUMMY_VEC4,
        } };
    } particles;

    struct {
        vks::Buffer fire;
        vks::Buffer environment;
    } uniformData;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec2 viewportDim;
        float pointSize = PARTICLE_SIZE;
    } uboVS;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 normal;
        glm::vec4 lightPos = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        glm::vec4 cameraPos;
    } uboEnv;

    struct {
        vk::Pipeline particles;
        vk::Pipeline environment;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    std::vector<Particle> particleBuffer;

    VulkanExample() {
        camera.setRotation({ -15.0f, 45.0f, 0.0f });
        camera.dolly(-90.0f);
        title = "Vulkan Example - Particle system";
        zoomSpeed *= 1.5f;
        timerSpeed *= 8.0f;
        srand((uint32_t)time(NULL));
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        textures.particles.smoke.destroy();
        textures.particles.fire.destroy();
        textures.floor.colorMap.destroy();
        textures.floor.normalMap.destroy();

        device.destroy(pipelines.particles);
        device.destroy(pipelines.environment);

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        particles.buffer.destroy();
        uniformData.fire.destroy();
        uniformData.environment.destroy();

        meshes.environment.destroy();
        device.destroy(textures.particles.sampler);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        // Environment
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.environment);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, meshes.descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.environment.vertices.buffer, vk::DeviceSize());
        cmdBuffer.bindIndexBuffer(meshes.environment.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.environment.indexCount, 1, 0, 0, 0);

        // Particle system
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.particles);
        cmdBuffer.bindVertexBuffers(0, particles.buffer.buffer, { 0 });
        cmdBuffer.draw(PARTICLE_COUNT, 1, 0, 0);
    }

    float rnd(float range) { return range * (rand() / float(RAND_MAX)); }

    void initParticle(Particle* particle, glm::vec3 emitterPos) {
        particle->vel = glm::vec4(0.0f, minVel.y + rnd(maxVel.y - minVel.y), 0.0f, 0.0f);
        particle->alpha = rnd(0.75f);
        particle->size = 1.0f + rnd(0.5f);
        particle->color = glm::vec4(1.0f);
        particle->type = PARTICLE_TYPE_FLAME;
        particle->rotation = rnd(2.0f * (float)M_PI);
        particle->rotationSpeed = rnd(2.0f) - rnd(2.0f);

        // Get random sphere point
        float theta = rnd(2 * (float)M_PI);
        float phi = rnd((float)M_PI) - (float)M_PI / 2;
        float r = rnd(FLAME_RADIUS);

        particle->pos.x = r * cos(theta) * cos(phi);
        particle->pos.y = r * sin(phi);
        particle->pos.z = r * sin(theta) * cos(phi);

        particle->pos += emitterPos;
    }

    void transitionParticle(Particle* particle) {
        switch (particle->type) {
            case PARTICLE_TYPE_FLAME:
                // Flame particles have a chance of turning into smoke
                if (rnd(1.0f) < 0.05f) {
                    particle->alpha = 0.0f;
                    particle->color = glm::vec4(0.25f + rnd(0.25f));
                    particle->pos.x *= 0.5f;
                    particle->pos.z *= 0.5f;
                    particle->vel = glm::vec4(rnd(1.0f) - rnd(1.0f), (minVel.y * 2) + rnd(maxVel.y - minVel.y), rnd(1.0f) - rnd(1.0f), 0.0f);
                    particle->size = 1.0f + rnd(0.5f);
                    particle->rotationSpeed = rnd(1.0f) - rnd(1.0f);
                    particle->type = PARTICLE_TYPE_SMOKE;
                } else {
                    initParticle(particle, emitterPos);
                }
                break;
            case PARTICLE_TYPE_SMOKE:
                // Respawn at end of life
                initParticle(particle, emitterPos);
                break;
        }
    }

    void prepareParticles() {
        particleBuffer.resize(PARTICLE_COUNT);
        for (auto& particle : particleBuffer) {
            initParticle(&particle, emitterPos);
            particle.alpha = 1.0f - (abs(particle.pos.y) / (FLAME_RADIUS * 2.0f));
        }

        particles.buffer = vks::Buffer::Builder{ sizeof(Particle) * PARTICLE_COUNT }
                               .withBufferUsage(vk::BufferUsageFlagBits::eVertexBuffer)
                               .withAllocCreateFlags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT)
                               .build();

        particles.buffer.map();
        particles.buffer.copy(particleBuffer);
    }

    void updateParticles() {
        float particleTimer = frameTimer * 0.45f;
        for (auto& particle : particleBuffer) {
            switch (particle.type) {
                case PARTICLE_TYPE_FLAME:
                    particle.pos.y -= particle.vel.y * particleTimer * 3.5f;
                    particle.alpha += particleTimer * 2.5f;
                    particle.size -= particleTimer * 0.5f;
                    break;
                case PARTICLE_TYPE_SMOKE:
                    particle.pos -= particle.vel * frameTimer * 1.0f;
                    particle.alpha += particleTimer * 1.25f;
                    particle.size += particleTimer * 0.125f;
                    particle.color -= particleTimer * 0.05f;
                    break;
            }
            particle.rotation += particleTimer * particle.rotationSpeed;
            // Transition particle state
            if (particle.alpha > 2.0f) {
                transitionParticle(&particle);
            }
        }

        particles.buffer.copy(particleBuffer);
    }

    void loadAssets() override {
        meshes.environment.loadFromFile(getAssetPath() + "models/fireplace.obj", meshes.vertexLayout, 10.0f);

        // Floor
        textures.floor.colorMap.loadFromFile(getAssetPath() + "textures/fireplace_colormap_bc3_unorm.ktx", vk::Format::eBc3UnormBlock);
        textures.floor.normalMap.loadFromFile(getAssetPath() + "textures/fireplace_normalmap_bc3_unorm.ktx", vk::Format::eBc3UnormBlock);

        // Particles
        textures.particles.smoke.loadFromFile(getAssetPath() + "textures/particle_smoke.ktx", vk::Format::eB8G8R8A8Unorm);
        textures.particles.fire.loadFromFile(getAssetPath() + "textures/particle_fire.ktx", vk::Format::eB8G8R8A8Unorm);

        // Create a custom sampler to be used with the particle textures
        // Create sampler
        vk::SamplerCreateInfo samplerCreateInfo;
        samplerCreateInfo.magFilter = vk::Filter::eLinear;
        samplerCreateInfo.minFilter = vk::Filter::eLinear;
        samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        // Different address mode
        samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eClampToBorder;
        samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
        samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.compareOp = vk::CompareOp::eNever;
        samplerCreateInfo.minLod = 0.0f;
        // Both particle textures have the same number of mip maps
        samplerCreateInfo.maxLod = (float)textures.particles.fire.image.createInfo.mipLevels;
        // Enable anisotropic filtering
        samplerCreateInfo.maxAnisotropy = 8;
        samplerCreateInfo.anisotropyEnable = VK_TRUE;
        // Use a different border color (than the normal texture loader) for additive blending
        samplerCreateInfo.borderColor = vk::BorderColor::eFloatTransparentBlack;
        textures.particles.sampler = device.createSampler(samplerCreateInfo);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes = { vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
                                                          vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 4) };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader image sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 1 : Fragment shader image sampler
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSets() {
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };
        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        // vk::Image descriptor for the color map texture
        vk::DescriptorImageInfo texDescriptorSmoke{ textures.particles.sampler, textures.particles.smoke.imageView, vk::ImageLayout::eReadOnlyOptimal };
        vk::DescriptorImageInfo texDescriptorFire{ textures.particles.sampler, textures.particles.fire.imageView, vk::ImageLayout::eReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            vk::WriteDescriptorSet{ descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.fire.descriptor },
            // Binding 1 : Smoke texture
            vk::WriteDescriptorSet{ descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorSmoke },
            // Binding 1 : Fire texture array
            vk::WriteDescriptorSet{ descriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorFire },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        // Environment
        meshes.descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        vk::DescriptorImageInfo texDescriptorColorMap{ defaultSampler, textures.floor.colorMap.imageView, vk::ImageLayout::eReadOnlyOptimal };
        vk::DescriptorImageInfo texDescriptorNormalMap{ defaultSampler, textures.floor.normalMap.imageView, vk::ImageLayout::eReadOnlyOptimal };

        writeDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            vk::WriteDescriptorSet{ meshes.descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.environment.descriptor },
            // Binding 1 : Color map
            vk::WriteDescriptorSet{ meshes.descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorColorMap },
            // Binding 2 : Normal map
            vk::WriteDescriptorSet{ meshes.descriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorNormalMap },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Environment rendering pipeline (normal mapped)
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        meshes.vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        pipelineBuilder.loadShader(vkx::shaders::particlefire::normalmap::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::particlefire::normalmap::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.environment = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Particle pipeline, read depth, but do not write it.
        // Premulitplied alpha
        pipelineBuilder.inputAssemblyState.topology = vk::PrimitiveTopology::ePointList;
        pipelineBuilder.depthStencilState.depthWriteEnable = VK_FALSE;
        auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[0];
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
        blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
        // Reset the vertex input state
        pipelineBuilder.vertexInputState = {};
        particles.vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        // Load shaders
        pipelineBuilder.loadShader(vkx::shaders::particlefire::particle::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::particlefire::particle::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.particles = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.fire = loader.createUniformBuffer(uboVS);
        // Vertex shader uniform buffer block
        uniformData.environment = loader.createUniformBuffer(uboEnv);

        updateUniformBuffers();
    }

    void updateUniformBufferLight() {
        // Environment
        uboEnv.lightPos.x = sin(timer * 2 * (float)M_PI) * 1.5f;
        uboEnv.lightPos.y = 0.0f;
        uboEnv.lightPos.z = cos(timer * 2 * (float)M_PI) * 1.5f;
        uniformData.environment.copy(uboEnv);
    }

    void updateUniformBuffers() {
        // Vertex shader
        glm::mat4 viewMatrix = glm::mat4();
        uboVS.projection = camera.matrices.perspective;
        uboVS.model = camera.matrices.view;
        uboVS.viewportDim = glm::vec2(size.width, size.height);
        uniformData.fire.copy(uboVS);

        // Environment
        uboEnv.projection = uboVS.projection;
        uboEnv.model = uboVS.model;
        uboEnv.normal = glm::inverseTranspose(uboEnv.model);
        uboEnv.cameraPos = glm::vec4(0.0, 0.0, camera.position.z, 0.0);
        uniformData.environment.copy(uboEnv);
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareParticles();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
        if (!paused) {
            updateUniformBufferLight();
            updateParticles();
        }
    }

    void viewChanged() override { updateUniformBuffers(); }
};

RUN_EXAMPLE(VulkanExample)
