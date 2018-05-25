/**
 * \file
 *
 * \brief The CPU version of the API.
 */
#pragma once

/// The cpu namespace houses all host side utilities.
namespace cpu {
    /**
     * \brief Filter a ``width`` x ``height`` image stored in row major order.
     *
     * \param width
     *     The width of the input image.
     *
     * \param height
     *     The height of the input image.
     *
     * \param input
     *     The input image to be filtered.  Assumed to already be allocated and
     *     be of size ``width * height * sizeof(float)``.
     *
     * \param output
     *     Where to store the filtered results.  Assumed to already be allocated
     *     and be of size ``width * height * sizeof(float)``.
     *
     * \param serial
     *     Whether or not the computation should be performed serially.  Default
     *     is ``false``, meaning the computation will be performed in parallel.
     */
    void filter(
        const int &width,
        const int &height,
        const float *input,
        float *output,
        bool serial = false
    );
}
