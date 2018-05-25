# -*- coding: utf8 -*-
########################################################################################
# This file is part of exhale.  Copyright (c) 2017-2018, Stephen McDowell.             #
# Full BSD 3-Clause license available here:                                            #
#                                                                                      #
#                https://github.com/svenevs/exhale/blob/master/LICENSE                 #
########################################################################################
"""
Tests for the ``cpp_cuda_mixed`` project.
"""

from __future__ import unicode_literals
import os
import pytest
import re
from sphinx.errors import ConfigError
import textwrap

from testing.base import ExhaleTestCase
from testing.decorators import confoverrides
from testing import hierarchies
from testing.hierarchies import                                       \
    class_hierarchy, compare_class_hierarchy, compare_file_hierarchy, \
    directory, file, file_hierarchy, function, namespace, signature


class CPPCudaMixed(ExhaleTestCase):
    """
    Primary test class for project ``cpp_cuda_mixed``.

    This test class exists to test explicit language lexer mappings via
    :data:`exhale.configs.lexerMapping`.
    """

    test_project = "cpp_cuda_mixed"
    """.. testproject:: cpp_cuda_mixed"""

    cuda_doxygen_input = textwrap.dedent('''
        INPUT              = ../include
        FILE_PATTERNS      = *.hpp *.cuh
        EXTENSION_MAPPING += cuh=c++
    ''')
    """Inform Doxygen to parse CUDA header files as C++."""

    def test_no_cuda(self):
        """
        Validate that ``include/declare.cuh`` is not included.

        Doxygen by default does not recognize ``.cuh`` files, this test examines
        that nothing unexpected shows up in the hierarchies.
        """
        self.checkRequiredConfigs()
        class_hierarchy_dict = {
            namespace("cpu"): {
                function("void", "filter"): signature(
                    "const int &width",
                    "const int &height",
                    "const float *input",
                    "float *output",
                    "bool serial = false"
                )
            }
        }
        compare_class_hierarchy(self, class_hierarchy(class_hierarchy_dict))

        file_hierarchy_dict = {
            directory("include"): {
                file("declare.hpp"): {
                    namespace("cpu"): {
                        function("void", "filter"): signature(
                            "const int &width",
                            "const int &height",
                            "const float *input",
                            "float *output",
                            "bool serial = false"
                        )
                    }
                }
            }
        }
        compare_file_hierarchy(self, file_hierarchy(file_hierarchy_dict))

    def get_declare_file_nodes(self, exhale_root):
        """
        Return the two file nodes associated with this project.

        **Parameters**
            ``exhale_root`` (:class:`exhale.graph.ExhaleRoot`)
                The graph root object to extract the ``include/declare.hpp`` and
                ``include/declare.cuh`` file nodes from.

        **Return**
            :class:`python:tuple` of :class:`exhale.graph.ExhaleNode`
                A length two tuple: ``(declare_hpp, declare_cuh)``.

        **Raises**
            :class:`python:RuntimeError`
                If either ``declare.hpp`` or ``declare.cuh`` cannot be found.
        """
        declare_hpp = None
        declare_cuh = None
        for f in exhale_root.files:
            if f.location == "include/declare.hpp":
                declare_hpp = f
            elif f.location == "include/declare.cuh":
                declare_cuh = f

        if not declare_hpp:
            raise RuntimeError(
                "Could not extract include/declare.hpp from exhale root..."
            )
        if not declare_cuh:
            raise RuntimeError(
                "Could not extract include/declare.cuh from exhale root..."
            )

        return declare_hpp, declare_cuh

    def validate_pygments_lexers(self, exhale_root, node_map):
        """
        Validate nodes have the expected pygments lexer in their program listing file.

        **Parameters**
            ``exhale_root`` (:class:`exhale.graph.ExhaleRoot`)
                The graph root object the keys in ``node_map`` were extracted from
                (e.g., by calling :func:`CPPCudaMixed.get_declare_file_nodes`).

            ``node_map`` (:class:`python:dict`)
                A map of :class:`exhale.graph.ExhaleNode` objects to string values.  The
                keys must be the nodes extracted from the ``exhale_root`` (as opposed
                to a testing type).  Each value should be a pygments lexer, such as
                ``"cpp"`` or ``"cuda"``.

                The generated program listing file will be parsed and an assert
                statement performed for equality to these specified values.
        """
        lexer_regex = re.compile(r"^.. code-block:: (.*)$")

        for file_node in node_map:
            expected_lexer = node_map[file_node]
            program_listing_file_path = os.path.join(
                exhale_root.root_directory,
                file_node.program_file
            )

            with open(program_listing_file_path) as program_listing_file:
                for line in program_listing_file:
                    lexer_match = lexer_regex.match(line)
                    if lexer_match:
                        lexer = lexer_match.groups()[0]
                        self.assertTrue(
                            lexer == expected_lexer,
                            "{0}: expected '{1}' but got '{2}' language lexer.".format(
                                file_node.location, expected_lexer, lexer
                            )
                        )
                        break

    @confoverrides(exhale_args={"exhaleDoxygenStdin": cuda_doxygen_input})
    def test_no_lang_lex(self):
        """
        Test that the CUDA code gets parsed.

        This test function validates the updated class and file hierarchies, and
        also that the CUDA file ``include/declare.cuh`` gets its program listing
        set to ``cpp`` (since that is what Doxygen was told to do).
        """
        self.checkRequiredConfigs()

        # now that Doxygen will parse include/declare.cuh, make sure the class
        # hierarchy contains the extra members
        class_hierarchy_dict = {
            namespace("cpu"): {
                function("void", "filter"): signature(
                    "const int &width",
                    "const int &height",
                    "const float *input",
                    "float *output",
                    "bool serial = false"
                )
            },
            namespace("cuda"): {
                function("void", "filter"): signature(
                    "const int &width",
                    "const int &height",
                    "const float *d_input",
                    "float *d_output",
                    "cudaStream_t stream = nullptr"
                ),
                namespace("kernels"): {
                    function("__global__ void", "filter"): signature(
                        "int width",
                        "int height",
                        "const float *d_input",
                        "float *d_output"
                    )
                }
            }
        }
        compare_class_hierarchy(self, class_hierarchy(class_hierarchy_dict))

        # similarly, make sure the file hierarchy checks out
        file_hierarchy_dict = {
            directory("include"): {
                file("declare.hpp"): {
                    namespace("cpu"): {
                        function("void", "filter"): signature(
                            "const int &width",
                            "const int &height",
                            "const float *input",
                            "float *output",
                            "bool serial = false"
                        )
                    }
                },
                file("declare.cuh"): {
                    namespace("cuda"): {
                        function("void", "filter"): signature(
                            "const int &width",
                            "const int &height",
                            "const float *d_input",
                            "float *d_output",
                            "cudaStream_t stream = nullptr"
                        ),
                        namespace("kernels"): {
                            function("__global__ void", "filter"): signature(
                                "int width",
                                "int height",
                                "const float *d_input",
                                "float *d_output"
                            )
                        }
                    }
                }
            }
        }
        compare_file_hierarchy(self, file_hierarchy(file_hierarchy_dict))

        # For this test, we told Doxygen to parse the .cuh file, and chose its
        # language to be C++ (via EXTENSION_MAPPING).  So as far as both Doxygen
        # and Exhale are concerned, both language lexers should be cpp.
        #
        # Only when Exhale is told to treat it as a different lexer will this
        # change (see test_with_cuda_with_lang_lex_glob).
        exhale_root = hierarchies._get_exhale_root(self)
        declare_hpp, declare_cuh = self.get_declare_file_nodes(exhale_root)
        import ipdb
        ipdb.set_trace()
        self.validate_pygments_lexers(
            exhale_root,
            {declare_hpp: "cpp", declare_cuh: "cpp"}
        )

    @confoverrides(exhale_args={
        "exhaleDoxygenStdin": cuda_doxygen_input,
        "lexerMapping": {r".*\.cuh": "cuda"}
    })
    def test_lang_lex_glob(self):
        r"""
        Test that ``"lexerMapping"`` overrides for globbed ``.*\.cuh``.
        """
        exhale_root = hierarchies._get_exhale_root(self)
        declare_hpp, declare_cuh = self.get_declare_file_nodes(exhale_root)
        self.validate_pygments_lexers(
            exhale_root,
            {declare_hpp: "cpp", declare_cuh: "cuda"}
        )

    @confoverrides(exhale_args={
        "exhaleDoxygenStdin": cuda_doxygen_input,
        "lexerMapping": {r"include/declare\.cuh": "cuda"}
    })
    def test_lang_lex_explicit(self):
        r"""
        Test that ``"lexerMapping"`` overrides for exact path to file location.
        """
        exhale_root = hierarchies._get_exhale_root(self)
        declare_hpp, declare_cuh = self.get_declare_file_nodes(exhale_root)
        self.validate_pygments_lexers(
            exhale_root,
            {declare_hpp: "cpp", declare_cuh: "cuda"}
        )

    @confoverrides(exhale_args={
        "exhaleDoxygenStdin": cuda_doxygen_input,
        "lexerMapping": {r"include/declare\.cuh": "cuda"}
    })
    def test_lang_lex_multi(self):
        """
        Test that multiple regular expressions will indeed be used.
        """
        exhale_root = hierarchies._get_exhale_root(self)
        declare_hpp, declare_cuh = self.get_declare_file_nodes(exhale_root)
        self.validate_pygments_lexers(
            exhale_root,
            {declare_hpp: "fortran", declare_cuh: "cuda"}
        )

    # # @pytest.mark.xfail()
    # # @pytest.mark.raises(exception=ConfigError)
    # @confoverrides(exhale_args={
    #     "exhaleDoxygenStdin": cuda_doxygen_input,
    #     "lexerMapping": {r"*\.cuh": "cuda"}
    # })
    # def test_bad_regex(self):
    #     """
    #     Verify that specifying invalid regular expression fails.

    #     .. todo:: figure out how to ``@pytest.mark.raises`` here instead of ``xfail``.
    #     """
    #     pass
