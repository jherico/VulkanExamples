//
//  Copyright � 2023 Bradley Austin Davis
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//

#pragma once
#include <vulkan/vulkan.hpp>
#include <functional>
#include <queue>

namespace vks {

///////////////////////////////////////////////////////////////////////
//
// Object destruction support
//
// It's often critical to avoid destroying an object that may be in use by the GPU.  In order to service this need
// the context class contains structures for objects that are pending deletion.
//
// The first container is the dumpster, and it just contains a set of lambda objects that when executed, destroy
// resources (presumably... in theory the lambda can do anything you want, but the purpose is to contain GPU object
// destruction calls, usually `vk::Device::destroy`).
//
// When the application renders a frame it can provide a fence along with the final command buffer of that frame.
// Once this fence is signaled, any items queued for destruction prior to that fence should be safe to delete. The
// fence is given to the recycler along with the dumpster contents.  The recycler then checks for any fences that
// have been signalled already (using a zero timeout) and executes the associated lambdas if it is.
//
// Finally, an application can call the recycle function at regular intervals (perhaps once per frame, perhaps less often)
// in order to check the fences and execute the associated destructors for any that are signalled.
struct Recycler {
private:
    Recycler() = default;
    Recycler(const Recycler&) = delete;
    Recycler& operator = (const Recycler&) = delete;

public:
    static Recycler& get();

    class Item {
    public:
        using Ptr = std::unique_ptr<Item>;
        virtual ~Item() = default;
        virtual void destroy() = 0;
    };

    template <typename T>
    class TypedItem : public Item{
    public:
        using Lambda = std::function<void(T&)>;
        TypedItem() = delete;
        TypedItem(const TypedItem&) = delete;
        TypedItem& operator=(const TypedItem&) = delete;
        TypedItem(T&& value, Lambda dtor)
            : value{ std::move(value) }
            , destructor{ std::move(dtor) } {}

        void destroy() override  { destructor(value); }

        T value;
        Lambda destructor;
    };

    // Template class to wrap vectors of objects
    template <typename T>
    struct TypedItemVector : public Item {
    public:
        using Lambda = std::function<void(const std::vector<T>&)>;

        TypedItemVector() = delete;
        TypedItemVector(const TypedItemVector&) = delete;
        TypedItemVector& operator=(const TypedItemVector&) = delete;

        TypedItemVector(std::vector<T>&& objs, Lambda dtor)
            : objects{ std::move(objs) }
            , destructor(std::move(dtor)) {}

        void destroy() override { destructor(objects); }

        std::vector<T> objects;
        Lambda destructor;
    };

    //using VoidLambdaList = std::list<VoidLambda>;
    using ItemList = std::list<Item::Ptr>;

    // A collection of items queued for destruction.  Once a fence has been created
    // for a queued submit, these items can be moved to the recycler for actual destruction
    // by calling the rec
    mutable ItemList dumpster;

    using FencedItem = std::pair<::vk::Fence, std::unique_ptr<Item>>;
    using FencedItemQueue = std::queue<FencedItem>;
    FencedItemQueue recycler;

    template <typename T>
    void trash(T&& value) const {
        trash<T>(std::move(value), [](T& t) { t.destroy(); });
    }

    template <typename T>
    void trash(T&& value, TypedItem<T>::Lambda destructor) const {
        if (!value) {
            return;
        }
        Item::Ptr ptr = std::make_unique<TypedItem<T>>(std::move(value), std::move(destructor));
        dumpster.push_back(std::move(ptr));
    }

    template <typename T>
    void trashAll(std::vector<T>&& values, TypedItemVector<T>::Lambda destructor) const {
        if (values.empty()) {
            return;
        }
        Item::Ptr ptr = std::make_unique<TypedItemVector<T>>(std::move(values), std::move(destructor));
        dumpster.push_back(std::move(ptr));
    }

    void trashCommandBuffers(const vk::CommandPool& commandPool, vk::CommandBuffer& cmdBuffer) const;
    void trashCommandBuffers(const vk::CommandPool& commandPool, std::vector<vk::CommandBuffer>& cmdBuffers) const;

    // Should be called from time to time by the application to migrate zombie resources
    // to the recycler along with a fence that will be signalled when the objects are
    // safe to delete.
    void emptyDumpster(vk::Fence fence);

    // Check the recycler fences for signalled status.  Any that are signalled will have their corresponding
    // lambdas executed, freeing up the associated resources
    void recycle();

    void flush();
};

}  // namespace vks