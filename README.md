# PanoViewSynthesis

* Code Assets
  * viewsynthesis.html : HTML/JS code for view synthesis
  * sample.js : JS code for sample 3d cube view from three.js docs
  * html_viewer.html : HTML code used by single-view MPI paper to render views
  
* Other
  * layers/ : image layers at different depths rendered by single-view MPI paper code
  * js/ : three.js assets 
  
To view in browser (Firefox is optimal) :

1. Run `python3 -m http.server` in this directory to start server (Python 3.x+)
2. Navigate to `http://0.0.0.0:8000/viewsynthesis.html`


* Unity Scripts
  * ProMesh.cs : Procedurally generates cylinder objects and materials using `Skybox/PanoramicBeta` shaders. One public variable for the folder name of all textures (should be in `Resources/` folder within Unity Project... `Resources/layers/courtyard_0`)
  * PanoramicBeta.shader : Custom 'Skybox/PanoramicBeta' shader

Single View MPI Generation:
 * single_view_mpi.sh : Downloads required code assets
 * single_view_mpi.py : runs script, takes in padding, output dir, input dir, and output width/height
