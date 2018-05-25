/**
 * \file
 *
 * \brief The CUDA version of the API.
 */
#pragma once

/// The cuda namespace houses all host side utilities.
namespace cuda {
    /// The kernels namespace houses all primary ``__global__`` entry points.
    namespace kernels {
        // Don't know why you would ever do this in a header file or if you even
        // can...but lets just ignore that ;)
        /**
         * \brief The CUDA kernel for filtering a ``width`` x ``height`` image
         *        stored in row major order.
         *
         * Users are encouraged to call the \ref cuda::filter wrapper function
         * rather than launch this kernel directly.
         *
         * \param width
         *     The width of the input image.
         *
         * \param height
         *     The height of the input image.
         *
         * \param d_input
         *     The input image to be filtered.  Assumed to already be allocated
         *     **on the device** and be of size
         *     ``width * height * sizeof(float)``.
         *
         * \param d_output
         *     Where to store the filtered results.  Assumed to already be
         *     allocated **on the device** and be of size
         *     ``width * height * sizeof(float)``.
         */
        __global__
        void filter(
            int width,
            int height,
            const float *d_input,
            float *d_output
        );
    }
    /**
     * \brief Filter a ``width`` x ``height`` image stored in row major order.
     *
     * \param width
     *     The width of the input image.
     *
     * \param height
     *     The height of the input image.
     *
     * \param d_input
     *     The input image to be filtered.  Assumed to already be allocated **on
     *     the device** and be of size ``width * height * sizeof(float)``.
     *
     * \param d_output
     *     Where to store the filtered results.  Assumed to already be allocated
     *     **on the device** and be of size ``width * height * sizeof(float)``.
     *
     * \param stream
     *     If not ``nullptr``, specifies the CUDA stream to execute the kernel
     *     launch on.  Otherwise, the kernel will be launched on the default
     *     CUDA stream.
     */
    void filter(
        const int &width,
        const int &height,
        const float *d_input,
        float *d_output,
        cudaStream_t stream = nullptr
    );
}
