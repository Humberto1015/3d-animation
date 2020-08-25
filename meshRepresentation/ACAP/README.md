# ACAP representation
## Files
```
├── ACAP.cpp
├── ACAP.h
├── angleSolver.cpp
├── angleSolver.h
├── application.cpp
├── application.h
├── axisSolver.cpp
├── axisSolver.h
├── main.cpp
├── mesh.cpp
├── mesh.h
└── npy.hpp
```

## Compile
```
mkdir build
cd build
cmake ..
make
```

## Functionalities
- show animation
```
./ACAP_bin [path_to_reference_mesh] --animation
```
- reconstruct mesh from ACAP feature
```
./ACAP_bin [path_to_reference_mesh] --reconstruct [path_to_ACAP_feature]
```
- Generate ACAP features
```
./ACAP_bin [path_to_reference_mesh] --genACAP [mesh_location] [ACAP_location]
```