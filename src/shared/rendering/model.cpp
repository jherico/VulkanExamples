/*
 * Vulkan Model loader using ASSIMP
 *
 * Copyright(C) 2016-2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license(MIT) (http://opensource.org/licenses/MIT)
 */

#include "model.hpp"
#include <rendering/loader.hpp>

#include <common/filesystem.hpp>

#include <assimp/Importer.hpp>
#include <assimp/cimport.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

using namespace vks;
using namespace vks::model;

const int Model::defaultFlags =
    aiProcess_FlipWindingOrder | aiProcess_Triangulate | aiProcess_PreTransformVertices | aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals;

void Model::loadFromFile(const std::string& filename, const VertexLayout& layout_, const ModelCreateInfo& createInfo, const int flags) {
    static const auto& context = vks::Context::get();
    static const auto& device = context.device;
    static const auto& loader = vks::Loader::get();
    layout = layout_;
    scale = createInfo.scale;
    uvscale = createInfo.uvscale;
    center = createInfo.center;
    destroy();
    // device = context.device;

    Assimp::Importer importer;
    const aiScene* pScene;

    // Load file
    vks::file::withBinaryFileContents(filename, [flags, &pScene, &importer](const char* filename_, vks::file::Span span) {
        pScene = importer.ReadFileFromMemory(span.data(), span.size(), flags, filename_);
    });

    if (!pScene) {
        std::string error = importer.GetErrorString();
        throw std::invalid_argument(
            error +
            "\n\nThe file may be part of the additional asset pack.\n\nRun \"download_assets.py\" in the repository root to download the latest version.");
    }

    parts.clear();
    parts.resize(pScene->mNumMeshes);
    for (unsigned int i = 0; i < pScene->mNumMeshes; i++) {
        const aiMesh* paiMesh = pScene->mMeshes[i];
        parts[i] = {};
        parts[i].name = paiMesh->mName.C_Str();
        parts[i].vertexBase = vertexCount;
        parts[i].vertexCount = paiMesh->mNumVertices;
        vertexCount += paiMesh->mNumVertices;
    }

    onLoad(importer, pScene);

    std::vector<uint8_t> vertexBuffer;
    std::vector<uint32_t> indexBuffer;

    vertexCount = 0;
    indexCount = 0;

    // Load meshes
    for (unsigned int meshIndex = 0; meshIndex < pScene->mNumMeshes; meshIndex++) {
        auto& part = parts[meshIndex];
        const aiMesh* paiMesh = pScene->mMeshes[meshIndex];
        const auto& numVertices = pScene->mMeshes[meshIndex]->mNumVertices;
        for (unsigned int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
            appendVertex(vertexBuffer, pScene, meshIndex, vertexIndex);
        }

        dim.size = dim.max - dim.min;

        vertexCount += numVertices;
        part.indexBase = static_cast<uint32_t>(indexBuffer.size());
        for (unsigned int j = 0; j < paiMesh->mNumFaces; j++) {
            const aiFace& Face = paiMesh->mFaces[j];
            if (Face.mNumIndices != 3)
                continue;
            indexBuffer.push_back(part.indexBase + Face.mIndices[0]);
            indexBuffer.push_back(part.indexBase + Face.mIndices[1]);
            indexBuffer.push_back(part.indexBase + Face.mIndices[2]);
            part.indexCount += 3;
        }
        indexCount += part.indexCount;
    }

    vks::QueueManager queueManager{ device, context.queuesInfo.graphics };
    // Vertex buffer
    vertices = loader.stageToDeviceBuffer(queueManager, vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);
    // Index buffer
    indices = loader.stageToDeviceBuffer(queueManager, vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    queueManager.destroy();
};

void Model::appendVertex(std::vector<uint8_t>& outputBuffer, const aiScene* pScene, uint32_t meshIndex, uint32_t vertexIndex) {
    static const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);
    const aiMesh* paiMesh = pScene->mMeshes[meshIndex];
    const auto& j = vertexIndex;
    aiColor3D pColor(0.f, 0.f, 0.f);
    pScene->mMaterials[paiMesh->mMaterialIndex]->Get(AI_MATKEY_COLOR_DIFFUSE, pColor);
    const aiVector3D* pPos = &(paiMesh->mVertices[j]);
    const aiVector3D* pNormal = &(paiMesh->mNormals[j]);
    const aiVector3D* pTexCoord = (paiMesh->HasTextureCoords(0)) ? &(paiMesh->mTextureCoords[0][j]) : &Zero3D;
    const aiVector3D* pTangent = (paiMesh->HasTangentsAndBitangents()) ? &(paiMesh->mTangents[j]) : &Zero3D;
    const aiVector3D* pBiTangent = (paiMesh->HasTangentsAndBitangents()) ? &(paiMesh->mBitangents[j]) : &Zero3D;
    std::vector<float> vertexBuffer;
    glm::vec3 scaledPos{ pPos->x, -pPos->y, pPos->z };
    scaledPos *= scale;
    scaledPos += center;

    // preallocate float buffer with approximate size
    vertexBuffer.reserve(layout.components.size() * 4);
    for (auto& component : layout.components) {
        switch (component) {
            case VERTEX_COMPONENT_POSITION:
                vertexBuffer.push_back(scaledPos.x);
                vertexBuffer.push_back(scaledPos.y);
                vertexBuffer.push_back(scaledPos.z);
                break;
            case VERTEX_COMPONENT_NORMAL:
                vertexBuffer.push_back(pNormal->x);
                vertexBuffer.push_back(-pNormal->y);
                vertexBuffer.push_back(pNormal->z);
                break;
            case VERTEX_COMPONENT_UV:
                vertexBuffer.push_back(pTexCoord->x * uvscale.s);
                vertexBuffer.push_back(pTexCoord->y * uvscale.t);
                break;
            case VERTEX_COMPONENT_COLOR:
                vertexBuffer.push_back(pColor.r);
                vertexBuffer.push_back(pColor.g);
                vertexBuffer.push_back(pColor.b);
                break;
            case VERTEX_COMPONENT_TANGENT:
                vertexBuffer.push_back(pTangent->x);
                vertexBuffer.push_back(pTangent->y);
                vertexBuffer.push_back(pTangent->z);
                break;
            case VERTEX_COMPONENT_BITANGENT:
                vertexBuffer.push_back(pBiTangent->x);
                vertexBuffer.push_back(pBiTangent->y);
                vertexBuffer.push_back(pBiTangent->z);
                break;
            // Dummy components for padding
            case VERTEX_COMPONENT_DUMMY_INT:
            case VERTEX_COMPONENT_DUMMY_FLOAT:
                vertexBuffer.push_back(0.0f);
                break;
            case VERTEX_COMPONENT_DUMMY_INT4:
            case VERTEX_COMPONENT_DUMMY_UINT4:
            case VERTEX_COMPONENT_DUMMY_VEC4:
                vertexBuffer.push_back(0.0f);
                vertexBuffer.push_back(0.0f);
                vertexBuffer.push_back(0.0f);
                vertexBuffer.push_back(0.0f);
                break;
            default:
                throw new std::invalid_argument("Bad case");
        };
    }
    appendOutput(outputBuffer, vertexBuffer);

    dim.max = glm::max(scaledPos, dim.max);
    dim.min = glm::min(scaledPos, dim.min);
}
