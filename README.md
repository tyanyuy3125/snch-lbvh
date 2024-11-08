# SNCH-LBVH

A tiny and lightning-fast implementation of the *Spatialized Normal Cone Hierarchy* (SNCH) based on a *Linear BVH* (LBVH) framework for Monte Carlo PDE research. A native and faster alternative to the *[fcpw](https://github.com/rohan-sawhney/fcpw)* library on CUDA.

This library supports three types of geometry queries:

* Nearest primitive query
* Nearest silhouette query
* Ray intersection

These queries fully cover all the geometric query functionality required by the Walk-on-Stars estimator.

This library is used in the high-performance *wavefront*-style PDE solver in the original research *Path Guiding for Monte Carlo PDE Solvers* (https://arxiv.org/abs/2410.18944).

## Getting started

This is a header-only library. The fcpw submodule is only used for benchmarking. If you only want to use the functions of this library, just clone it (no recursive required) and add the include folder to your project.

`line_test.cu` and `triangle_test.cu` demonstrate the usage of this library.

## Performance

The following is a performance comparison of GPU query functions of this library (SNCH-LBVH) and *fcpw* (using Vulkan backend on Linux) under RTX 4090 GPU. 

**This library is faster than fcpw in all test cases and query types.**

![benchmark result](benchmark.png)

## Known Issues

* CPU query part does not work.
* Some functions have strange conversions from float3 to float4. The design here is not unified (considering the characteristics of GPU, all should be replaced with float4 in the future).

## Roadmap

* Fix known issues mentioned above.
* Consider memory locality.
* Support more `.obj` face types (currently only supports triangle faces).

## License and Citations

The very original code base is authored by [ToruNiina](https://github.com/ToruNiina/lbvh). Later, [rsugimoto](https://github.com/rsugimoto/lbvh) edits this library to support 2D primitives and intersection operations to support the Walk on Boundary research ([WoBToolbox](https://github.com/rsugimoto/WoBToolbox)). 

I make major changes to the entire project, introducing the closest silhouette query and the closest ray intersection query.

This library is used in the original research *Path Guiding for Monte Carlo PDE Solvers*, which builds a high-performance *wavefront*-style solver on GPU that requires an SNCH geometry query implementation on CUDA.

If you plan to use/modify this project, please cite relevant papers:

```bibtex
@misc{huang2024pathguidingpde,
      title={Path Guiding for Monte Carlo PDE Solvers}, 
      author={Tianyu Huang and Jingwang Ling and Shuang Zhao and Feng Xu},
      year={2024},
      eprint={2410.18944},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2410.18944}, 
}

@article{sugimoto2023wob,
   title={A Practical Walk-on-Boundary Method for Boundary Value Problems},
   volume={42},
   ISSN={1557-7368},
   url={http://dx.doi.org/10.1145/3592109},
   DOI={10.1145/3592109},
   number={4},
   journal={ACM Transactions on Graphics},
   publisher={Association for Computing Machinery (ACM)},
   author={Sugimoto, Ryusuke and Chen, Terry and Jiang, Yiti and Batty, Christopher and Hachisuka, Toshiya},
   year={2023},
   month=jul, pages={1–16}
}
```

Please keep the original `LICENSE` file from ToruNiina and give the [original project](https://github.com/ToruNiina/lbvh) a star.
