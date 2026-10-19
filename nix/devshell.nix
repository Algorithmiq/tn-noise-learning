{
  lib,
  mkShell,
  stdenv,
  python,
  git,
  just,
  lychee,
  uv,
  zlib,
}:

mkShell {
  nativeBuildInputs = [
    git
    just
    lychee
    python
    uv
  ];

  shellHook = ''
    # `[tool.uv] python-preference = "only-managed"` pulls prebuilt interpreters that expect a loader NixOS does not
    # provide; use this shell's CPython.
    export UV_PYTHON_PREFERENCE=only-system
    export UV_PYTHON="${python}/bin/python"

    # manylinux wheels (jaxlib, numpy, qiskit, ...) resolve libstdc++ through the ambient loader path, which is empty
    # on NixOS.
    export LD_LIBRARY_PATH="${
      lib.makeLibraryPath [
        stdenv.cc.cc.lib
        zlib
      ]
    }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  '';
}
