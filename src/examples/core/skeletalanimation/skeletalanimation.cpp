/*
 * Vulkan Example - Skeletal animation
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>
#include <examples/keycodes.hpp>

#include <assimp/Importer.hpp>
#include <assimp/anim.h>
#include <assimp/matrix4x4.h>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <shaders/skeletalanimation/mesh.frag.inl>
#include <shaders/skeletalanimation/mesh.vert.inl>
#include <shaders/skeletalanimation/texture.frag.inl>
#include <shaders/skeletalanimation/texture.vert.inl>

// Vertex layout used in this example
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 color;
    // Max. four bones per vertex
    float boneWeights[4];
    uint32_t boneIDs[4];
};

vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_UV,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
    vks::model::Component::VERTEX_COMPONENT_DUMMY_VEC4,
    vks::model::Component::VERTEX_COMPONENT_DUMMY_UINT4,
} };

// Maximum number of bones per mesh
// Must not be higher than same const in skinning shader
#define MAX_BONES 64
// Maximum number of bones per vertex
#define MAX_BONES_PER_VERTEX 4

// Skinned mesh class

// Per-vertex bone IDs and weights
struct VertexBoneData {
    std::array<uint32_t, MAX_BONES_PER_VERTEX> IDs;
    std::array<float, MAX_BONES_PER_VERTEX> weights;

    // Ad bone weighting to vertex info
    void add(uint32_t boneID, float weight) {
        for (uint32_t i = 0; i < MAX_BONES_PER_VERTEX; i++) {
            if (weights[i] == 0.0f) {
                IDs[i] = boneID;
                weights[i] = weight;
                return;
            }
        }
    }
};

// Stores information on a single bone
struct BoneInfo {
    aiMatrix4x4 offset;
    aiMatrix4x4 finalTransformation;

    BoneInfo() {
        offset = aiMatrix4x4();
        finalTransformation = aiMatrix4x4();
    };
};

class SkinnedMesh : public vks::model::Model {
public:
    // Bone related stuff
    // Maps bone name with index
    std::unordered_map<std::string, uint32_t> boneMapping;
    // Bone details
    std::vector<BoneInfo> boneInfo;
    // Number of bones present
    uint32_t numBones = 0;
    // Root inverese transform matrix
    aiMatrix4x4 globalInverseTransform;
    // Per-vertex bone info
    std::vector<VertexBoneData> bones;
    // Bone transformations
    std::vector<aiMatrix4x4> boneTransforms;

    // Modifier for the animation
    float animationSpeed = 0.75f;
    // Currently active animation
    const aiScene* pScene{ nullptr };
    aiAnimation* pAnimation{ nullptr };
    uint32_t numAnimations{ 0 };

    // Vulkan buffers
    vks::model::Model meshBuffer;
    // Reference to assimp mesh
    // Required for animation

    void onLoad(Assimp::Importer& importer, const aiScene* pScene) override {
        this->pScene = importer.GetOrphanedScene();
        // Setup bones
        // One vertex bone info structure per vertex
        bones.resize(vertexCount);
        numAnimations = pScene->mNumAnimations;
        // Store global inverse transform matrix of root node
        globalInverseTransform = pScene->mRootNode->mTransformation;
        globalInverseTransform.Inverse();
        // Load bones (weights and IDs)
        for (uint32_t m = 0; m < pScene->mNumMeshes; m++) {
            aiMesh* paiMesh = pScene->mMeshes[m];
            if (paiMesh->mNumBones > 0) {
                loadBones(m, paiMesh, bones);
            }
        }
    }

    void appendVertex(std::vector<uint8_t>& outputBuffer, const aiScene* pScene, uint32_t meshIndex, uint32_t vertexIndex) override {
        const auto& part = parts[meshIndex];
        const auto& bone = bones[part.vertexBase + vertexIndex];
        const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);
        const aiMesh* paiMesh = pScene->mMeshes[meshIndex];
        const aiVector3D* pPos = &(paiMesh->mVertices[vertexIndex]);
        const aiVector3D* pNormal = &(paiMesh->mNormals[vertexIndex]);
        const aiVector3D* pTexCoord = (paiMesh->HasTextureCoords(0)) ? &(paiMesh->mTextureCoords[0][vertexIndex]) : &Zero3D;
        const aiVector3D* pTangent = (paiMesh->HasTangentsAndBitangents()) ? &(paiMesh->mTangents[vertexIndex]) : &Zero3D;
        const aiVector3D* pBiTangent = (paiMesh->HasTangentsAndBitangents()) ? &(paiMesh->mBitangents[vertexIndex]) : &Zero3D;

        aiColor3D pColor(0.f, 0.f, 0.f);
        pScene->mMaterials[paiMesh->mMaterialIndex]->Get(AI_MATKEY_COLOR_DIFFUSE, pColor);

        Vertex vertex;
        vertex.pos = { pPos->x, -pPos->y, pPos->z };
        vertex.pos *= scale;
        vertex.pos += center;
        vertex.normal = { pNormal->x, -pNormal->y, pNormal->z };
        vertex.uv = { pTexCoord->x, pTexCoord->y };
        vertex.uv *= uvscale;
        vertex.color = { pColor.r, pColor.g, pColor.b };

        // Fetch bone weights and IDs
        for (uint32_t boneIndex = 0; boneIndex < MAX_BONES_PER_VERTEX; boneIndex++) {
            vertex.boneWeights[boneIndex] = bone.weights[boneIndex];
            vertex.boneIDs[boneIndex] = bone.IDs[boneIndex];
        }

        dim.max = glm::max(vertex.pos, dim.max);
        dim.min = glm::min(vertex.pos, dim.min);

        appendOutput(outputBuffer, vertex);
    }

    // Set active animation by index
    void setAnimation(uint32_t animationIndex) {
        assert(animationIndex < numAnimations);
        pAnimation = pScene->mAnimations[animationIndex];
    }

    // Load bone information from ASSIMP mesh
    void loadBones(uint32_t meshIndex, const aiMesh* pMesh, std::vector<VertexBoneData>& Bones) {
        for (uint32_t i = 0; i < pMesh->mNumBones; i++) {
            uint32_t index = 0;

            assert(pMesh->mNumBones <= MAX_BONES);

            std::string name(pMesh->mBones[i]->mName.data);

            if (boneMapping.find(name) == boneMapping.end()) {
                // Bone not present, add new one
                index = numBones;
                numBones++;
                BoneInfo bone;
                boneInfo.push_back(bone);
                boneInfo[index].offset = pMesh->mBones[i]->mOffsetMatrix;
                boneMapping[name] = index;
            } else {
                index = boneMapping[name];
            }

            for (uint32_t j = 0; j < pMesh->mBones[i]->mNumWeights; j++) {
                uint32_t vertexID = parts[meshIndex].vertexBase + pMesh->mBones[i]->mWeights[j].mVertexId;
                Bones[vertexID].add(index, pMesh->mBones[i]->mWeights[j].mWeight);
            }
        }
        boneTransforms.resize(numBones);
    }

    // Recursive bone transformation for given animation time
    void update(float time) {
        float TicksPerSecond = (float)(pScene->mAnimations[0]->mTicksPerSecond != 0 ? pScene->mAnimations[0]->mTicksPerSecond : 25.0f);
        float TimeInTicks = time * TicksPerSecond;
        float AnimationTime = fmod(TimeInTicks, (float)pScene->mAnimations[0]->mDuration);

        aiMatrix4x4 identity = aiMatrix4x4();
        readNodeHierarchy(AnimationTime, pScene->mRootNode, identity);

        for (uint32_t i = 0; i < boneTransforms.size(); i++) {
            boneTransforms[i] = boneInfo[i].finalTransformation;
        }
    }

private:
    // Find animation for a given node
    const aiNodeAnim* findNodeAnim(const aiAnimation* animation, const std::string nodeName) {
        for (uint32_t i = 0; i < animation->mNumChannels; i++) {
            const aiNodeAnim* nodeAnim = animation->mChannels[i];
            if (std::string(nodeAnim->mNodeName.data) == nodeName) {
                return nodeAnim;
            }
        }
        return nullptr;
    }

    // Returns a 4x4 matrix with interpolated translation between current and next frame
    aiMatrix4x4 interpolateTranslation(float time, const aiNodeAnim* pNodeAnim) {
        aiVector3D translation;

        if (pNodeAnim->mNumPositionKeys == 1) {
            translation = pNodeAnim->mPositionKeys[0].mValue;
        } else {
            uint32_t frameIndex = 0;
            for (uint32_t i = 0; i < pNodeAnim->mNumPositionKeys - 1; i++) {
                if (time < (float)pNodeAnim->mPositionKeys[i + 1].mTime) {
                    frameIndex = i;
                    break;
                }
            }

            aiVectorKey currentFrame = pNodeAnim->mPositionKeys[frameIndex];
            aiVectorKey nextFrame = pNodeAnim->mPositionKeys[(frameIndex + 1) % pNodeAnim->mNumPositionKeys];

            float delta = (time - (float)currentFrame.mTime) / (float)(nextFrame.mTime - currentFrame.mTime);

            const aiVector3D& start = currentFrame.mValue;
            const aiVector3D& end = nextFrame.mValue;

            translation = (start + delta * (end - start));
        }

        aiMatrix4x4 mat;
        aiMatrix4x4::Translation(translation, mat);
        return mat;
    }

    // Returns a 4x4 matrix with interpolated rotation between current and next frame
    aiMatrix4x4 interpolateRotation(float time, const aiNodeAnim* pNodeAnim) {
        aiQuaternion rotation;

        if (pNodeAnim->mNumRotationKeys == 1) {
            rotation = pNodeAnim->mRotationKeys[0].mValue;
        } else {
            uint32_t frameIndex = 0;
            for (uint32_t i = 0; i < pNodeAnim->mNumRotationKeys - 1; i++) {
                if (time < (float)pNodeAnim->mRotationKeys[i + 1].mTime) {
                    frameIndex = i;
                    break;
                }
            }

            aiQuatKey currentFrame = pNodeAnim->mRotationKeys[frameIndex];
            aiQuatKey nextFrame = pNodeAnim->mRotationKeys[(frameIndex + 1) % pNodeAnim->mNumRotationKeys];

            float delta = (time - (float)currentFrame.mTime) / (float)(nextFrame.mTime - currentFrame.mTime);

            const aiQuaternion& start = currentFrame.mValue;
            const aiQuaternion& end = nextFrame.mValue;

            aiQuaternion::Interpolate(rotation, start, end, delta);
            rotation.Normalize();
        }

        aiMatrix4x4 mat(rotation.GetMatrix());
        return mat;
    }

    // Returns a 4x4 matrix with interpolated scaling between current and next frame
    aiMatrix4x4 interpolateScale(float time, const aiNodeAnim* pNodeAnim) {
        aiVector3D scale;

        if (pNodeAnim->mNumScalingKeys == 1) {
            scale = pNodeAnim->mScalingKeys[0].mValue;
        } else {
            uint32_t frameIndex = 0;
            for (uint32_t i = 0; i < pNodeAnim->mNumScalingKeys - 1; i++) {
                if (time < (float)pNodeAnim->mScalingKeys[i + 1].mTime) {
                    frameIndex = i;
                    break;
                }
            }

            aiVectorKey currentFrame = pNodeAnim->mScalingKeys[frameIndex];
            aiVectorKey nextFrame = pNodeAnim->mScalingKeys[(frameIndex + 1) % pNodeAnim->mNumScalingKeys];

            float delta = (time - (float)currentFrame.mTime) / (float)(nextFrame.mTime - currentFrame.mTime);

            const aiVector3D& start = currentFrame.mValue;
            const aiVector3D& end = nextFrame.mValue;

            scale = (start + delta * (end - start));
        }

        aiMatrix4x4 mat;
        aiMatrix4x4::Scaling(scale, mat);
        return mat;
    }

    // Get node hierarchy for current animation time
    void readNodeHierarchy(float AnimationTime, const aiNode* pNode, const aiMatrix4x4& ParentTransform) {
        std::string NodeName(pNode->mName.data);

        aiMatrix4x4 NodeTransformation(pNode->mTransformation);

        const aiNodeAnim* pNodeAnim = findNodeAnim(pAnimation, NodeName);

        if (pNodeAnim) {
            // Get interpolated matrices between current and next frame
            aiMatrix4x4 matScale = interpolateScale(AnimationTime, pNodeAnim);
            aiMatrix4x4 matRotation = interpolateRotation(AnimationTime, pNodeAnim);
            aiMatrix4x4 matTranslation = interpolateTranslation(AnimationTime, pNodeAnim);

            NodeTransformation = matTranslation * matRotation * matScale;
        }

        aiMatrix4x4 GlobalTransformation = ParentTransform * NodeTransformation;

        if (boneMapping.find(NodeName) != boneMapping.end()) {
            uint32_t BoneIndex = boneMapping[NodeName];
            boneInfo[BoneIndex].finalTransformation = globalInverseTransform * GlobalTransformation * boneInfo[BoneIndex].offset;
        }

        for (uint32_t i = 0; i < pNode->mNumChildren; i++) {
            readNodeHierarchy(AnimationTime, pNode->mChildren[i], GlobalTransformation);
        }
    }
};

class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        vks::texture::Texture2D colorMap;
        vks::texture::Texture2D floor;
    } textures;

    SkinnedMesh skinnedMesh;

    struct {
        vks::Buffer vsScene;
        vks::Buffer floor;
    } uniformData;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 bones[MAX_BONES];
        glm::vec4 lightPos = glm::vec4(0.0f, -250.0f, 250.0f, 1.0);
        glm::vec4 viewPos;
    } uboVS;

    struct UboFloor {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(0.0, 0.0f, -25.0f, 1.0);
        glm::vec4 viewPos;
        glm::vec2 uvOffset;
    } uboFloor;

    struct {
        vk::Pipeline skinning;
        vk::Pipeline texture;
    } pipelines;

    struct {
        vks::model::Model floor;
    } meshes;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    struct {
        vk::DescriptorSet skinning;
        vk::DescriptorSet floor;
    } descriptorSets;

    float runningTime = 0.0f;

    VulkanExample() {
        camera.type = camera.lookat;
        zoomSpeed = 2.5f;
        rotationSpeed = 0.5f;
        camera.dolly(-150.0f);
        camera.setRotation({ -25.5f, 128.5f, 180.0f });
        title = "Vulkan Example - Skeletal animation";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.skinning);

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        textures.colorMap.destroy();

        uniformData.vsScene.destroy();

        // Destroy and free mesh resources
        skinnedMesh.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));

        // Skinned mesh
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skinning);
        cmdBuffer.bindVertexBuffers(0, skinnedMesh.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(skinnedMesh.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(skinnedMesh.indexCount, 1, 0, 0, 0);

        // Floor
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.floor, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.texture);
        cmdBuffer.bindVertexBuffers(0, meshes.floor.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.floor.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.floor.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override {
        textures.colorMap.loadFromFile(getAssetPath() + "textures/goblin_bc3.ktx", vk::Format::eBc3UnormBlock);
        textures.floor.loadFromFile(getAssetPath() + "textures/pattern_35_bc3.ktx", vk::Format::eBc3UnormBlock);
        meshes.floor.loadFromFile(getAssetPath() + "models/plane_z.obj", vertexLayout, 512.0f);
        // Load a mesh based on data read via assimp
        skinnedMesh.loadFromFile(getAssetPath() + "models/goblin.dae", vertexLayout, 1.0f,
                                 aiProcess_FlipWindingOrder | aiProcess_Triangulate | aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals);
        skinnedMesh.setAnimation(0);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one combined image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2),
        };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader combined sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };

        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        vk::DescriptorImageInfo texDescriptor{ defaultSampler, textures.colorMap.imageView, vk::ImageLayout::eReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsScene.descriptor },
            // Binding 1 : Color map
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        // Floor
        descriptorSets.floor = device.allocateDescriptorSets(allocInfo)[0];
        texDescriptor.imageView = textures.floor.imageView;
        texDescriptor.sampler = defaultSampler;
        writeDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSets.floor, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.floor.descriptor },
            // Binding 1 : Color map
            { descriptorSets.floor, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Skinned rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineCreator{ device, pipelineLayout };
        pipelineCreator.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineCreator.depthStencilState = true;
        vertexLayout.appendVertexLayout(pipelineCreator.vertexInputState);
        pipelineCreator.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineCreator.loadShader(vkx::shaders::skeletalanimation::mesh::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(vkx::shaders::skeletalanimation::mesh::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.skinning = pipelineCreator.create(context.pipelineCache);
        pipelineCreator.destroyShaderModules();
        pipelineCreator.loadShader(vkx::shaders::skeletalanimation::texture::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(vkx::shaders::skeletalanimation::texture::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.texture = pipelineCreator.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vsScene = loader.createUniformBuffer(uboVS);

        // Floor
        uniformData.floor = loader.createUniformBuffer(uboFloor);

        updateUniformBuffers(true);
    }

    void updateUniformBuffers(bool viewChanged) {
        if (viewChanged) {
            uboFloor.projection = uboVS.projection = getProjection();
            uboFloor.model = uboVS.model = glm::scale(glm::rotate(camera.matrices.view, glm::radians(90.0f), glm::vec3(1, 0, 0)), glm::vec3(0.025f));
            uboFloor.viewPos = uboVS.viewPos = glm::vec4(0.0f, 0.0f, -camera.position.z, 0.0f);
            uboFloor.model = glm::translate(uboFloor.model, glm::vec3(0.0f, 0.0f, -1800.0f));
        }

        // Update bones
        skinnedMesh.update(runningTime);
        for (uint32_t i = 0; i < skinnedMesh.boneTransforms.size(); i++) {
            uboVS.bones[i] = glm::transpose(glm::make_mat4(&skinnedMesh.boneTransforms[i].a1));
        }

        uniformData.vsScene.copy(uboVS);

        // Update floor animation
        uboFloor.uvOffset.t -= 0.5f * skinnedMesh.animationSpeed * frameTimer;
        uniformData.floor.copy(uboFloor);
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
        if (!paused) {
            runningTime += frameTimer * skinnedMesh.animationSpeed;
            updateUniformBuffers(false);
        }
    }

    void viewChanged() override { updateUniformBuffers(true); }

    void changeAnimationSpeed(float delta) {
        skinnedMesh.animationSpeed += delta;
        std::cout << "Animation speed = " << skinnedMesh.animationSpeed << std::endl;
    }

    void keyPressed(uint32_t key) override {
        switch (key) {
            case KEY_KPADD:
            case KEY_KPSUB:
                changeAnimationSpeed((key == KEY_KPADD) ? 0.1f : -0.1f);
                break;
        }
    }
};

RUN_EXAMPLE(VulkanExample)
