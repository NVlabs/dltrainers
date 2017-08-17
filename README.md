# Introduction

The `dlpipes` library is a set of modules for expressing input pipelines,
deep networks, and network training compactly and easily.

Right now, there isn't much documentation, but you can see examples of
the library in action in `intro-to-dlpipes.ipynb` (a Jupyter Notebook).

For creating sharded tarfiles, use the `tarshards` command.  The
following will find all `.png` and `.cls` files in the current
directory tree, tar them up in shards of size 1 Gbyte, and transfer
them via `ssh` to your home directory on host `mvcdev2` with names
like `myimages-000000.tgz`, ...:

```
find . -name '*.png' -o -name '*.cls' | tarshards -S mvcdev2: myimages
```

To create a shardindex, run `shardindex myimages` in the destination
directory (TODO: create and upload the shardindex automatically in
`tarshards`).

# TODO

Integrate the prototypes of the following iterators into `dlpipes`:

- itselect(...)
- itmerge(...)
- itvideos(...)
- itmodel(...)
- itmemcache(...)
- itdiskcache(...)
- itscache(...)

More:

- handle multiple images correctly in `itstandardize`
- add image warping
