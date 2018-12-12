#pragma once

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <set>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <chrono>
#include <unordered_map>
#include <glm/gtx/hash.hpp>
#include "Vertex.h"
#include "SwapChain.h"
#include "RenderPass.h"

namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

//descriptor used to load objs
struct UniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

class Application
{
public:
	Application();
	~Application();

	void run();

private:
	//GLFW window used to display application
	GLFWwindow* window;

	//Instance of Vulkan, connection between the application and Vulkan
	VkInstance instance;
	//callback to show debug messages
	VkDebugUtilsMessengerEXT callback;
	//the surface that will go over the window system in whatever platform is being used
	VkSurfaceKHR surface;

	//Graphics card we'll be using
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	//Logical device being used
	VkDevice device;

	//queue for graphics
	VkQueue graphicsQueue;
	//queue for presentation
	VkQueue presentQueue;

	//Swap Chain members
	SwapChain swapChainClass;

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	//view into the images
	std::vector<VkImageView> swapChainImageViews;

	RenderPass renderPassClass;
	VkRenderPass renderPass;
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;
	VkPipelineLayout pipelineLayout;

	VkPipeline graphicsPipeline;

	std::vector<VkFramebuffer> swapChainFramebuffers;

	//Command pool to manage the memory used to store the buffers and command buffers
	VkCommandPool commandPool;


	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;

	uint32_t mipLevels;
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;

	VkImage colorImage;
	VkDeviceMemory colorImageMemory;
	VkImageView colorImageView;

	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	size_t currentFrame = 0;

	bool framebufferResized = false;

	void initWindow();

	void initVulkan();

	void mainLoop();

	void drawFrame();

	void cleanupSwapChain();

	void cleanup();

	void createInstance();
	
	void setupDebugCallback();

	void createSurface();

	void pickPhysicalDevice();

	void createLogicalDevice();

	void createImageViews();

	void createRenderPass();

	void createDescriptorSetLayout();

	void createDescriptorPool();

	void createDescriptorSets();

	void createGraphicsPipeline();

	void createFramebuffers();

	void createCommandPool();

	void createColorResources();

	void createDepthResources();

	void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling,
		VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

	void createTextureImage();

	void createTextureImageView();

	void createTextureSampler();

	void loadModel();

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
		VkBuffer& buffer, VkDeviceMemory& bufferMemory);

	void createVertexBuffer();

	void createIndexBuffer();

	void createUniformBuffers();

	void updateUniformBuffer(uint32_t currentImage);

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

	void createCommandBuffers();

	void createSyncObjects();

	void recreateSwapChain();

	void updateSwapChain();

	std::vector<const char*> getRequiredExtensions();

	bool checkValidationLayerSupport();

	bool isDeviceSuitable(VkPhysicalDevice device);

	bool checkDeviceExtensionSupport(VkPhysicalDevice device);

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

	VkShaderModule createShaderModule(const std::vector<char>& code);

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes);

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

	//Callback function, or function the displays diagnostic messages
	//Takes in the level of severity, type of message, a struct of the details of the message,
	//and any data you would like to pass through
	static VKAPI_ATTR VkBool32 VKAPI_CALL Application::debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

	static std::vector<char> readFile(const std::string& fileName)
	{
		//sets flags to read at the end of the file and read in binary
		std::ifstream file(fileName, std::ios::ate | std::ios::binary);

		if (!file.is_open())
			throw std::runtime_error("failed to open file!");

		//read the end of the file to find the size of the file and allocate a buffer
		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);
		file.seekg(0);
		file.read(buffer.data(), fileSize);
		file.close();

		return buffer;
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	VkCommandBuffer beginSingleTimeCommands();

	void endSingleTimeCommands(VkCommandBuffer commandBuffer);

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout,
		VkImageLayout newLayout, uint32_t mipLevels);

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);

	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

	VkFormat findDepthFormat();

	bool hasStencilComponent(VkFormat format);

	VkSampleCountFlagBits getMaxUsableSampleCount();

	VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
		const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
		const VkAllocationCallbacks* pAllocator,
		VkDebugUtilsMessengerEXT* pCallback);

	void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback,
		const VkAllocationCallbacks* pAllocator);
};

