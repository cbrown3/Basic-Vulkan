#pragma once

#include <GLFW/glfw3.h>
#include <vector>
#include <optional>
#include <algorithm>

//Queue families are the group of commands needed to be submitted
//to a queue, which only allows a specific subset of commands
struct QueueFamilyIndices
{
	//checks if graphics and present commands can be done
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	//is true if the GPU can support both queue families
	bool isComplete()
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

//used to check if the swapchain support is good enough
struct SwapChainSupportDetails
{
	//the min/max num of images in swap chain, min/max width and height of images
	VkSurfaceCapabilitiesKHR capabilities;
	//surface formats i.e. pixel format, color space
	std::vector<VkSurfaceFormatKHR> formats;
	//available presentation modes
	std::vector<VkPresentModeKHR> presentModes;
};

class SwapChain
{
public:
	SwapChain();
	~SwapChain();

	void update(GLFWwindow* window, VkSurfaceKHR surface, VkPhysicalDevice physicalDevice, VkDevice device);
	void clear(VkDevice device);

	VkSwapchainKHR GetSwapChain() { return swapChain; }
	std::vector<VkImage> GetSwapChainImages() { return swapChainImages; }
	VkFormat GetSwapChainImageFormat() { return swapChainImageFormat; }
	VkExtent2D GetSwapChainExtent() { return swapChainExtent; }
	std::vector<VkImageView> GetSwapChainImageViews() { return swapChainImageViews; }


private:
	//Swap Chain members
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	//view into the images
	std::vector<VkImageView> swapChainImageViews;

	QueueFamilyIndices findQueueFamilies(VkSurfaceKHR surface, VkPhysicalDevice device);
	SwapChainSupportDetails querySwapChainSupport(VkSurfaceKHR surface, VkPhysicalDevice device);
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes);
	VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities);
};

