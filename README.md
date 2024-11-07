# SNCH-LBVH

A simple and tiny implementation of the *Spatialized normal cone hierarchy* (SNCH) based on a *Linear BVH* (LBVH) framework for Monte Carlo PDE research. A native alternative to the *[fcpw](https://github.com/rohan-sawhney/fcpw)* library on CUDA.

This library is used in the high-performance *wavefront*-style PDE solver in the original research *Path Guiding for Monte Carlo PDE Solvers* (https://arxiv.org/abs/2410.18944).

## Getting started

This is a header-only library. The fcpw submodule is only used for benchmarking. If you only want to use the functions of this library, just clone it (no recursive required) and add the include folder to your project.

`line_test.cu` and `triangle_test.cu` demonstrate the usage of this library.

## Performance

The current implementation is quick-and-dirty and does not fully consider the locality of memory access. Therefore, although this library performs significantly better than the fcpw library in 2D and 3D cases with a low number of faces, it performs weaker than the fcpw library when the GPU memory occupancy is large. This issue has been identified and will be resolved in a subsequent update.

![benchmark result](benchmark.png)

## Known Issues

I am still organizing the code of this library. In addition to performance issues, there are some known issues:

* CPU query part does not work.
* Some functions have strange conversions from float3 to float4. The design here is not unified (considering the characteristics of GPU, all should be replaced with float4 in the future).

## Roadmap

* Fix known issues mentioned above.
* More aggressive pruning strategies.
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
