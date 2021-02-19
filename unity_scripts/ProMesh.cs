using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ProMesh : MonoBehaviour
{
    public string folder;

    // Start is called before the first frame update
    void Start()
    {
        double[] depths = { 100,
                    23.846155,
                    13.537119,
                    9.45122,
                    7.259953,
                    5.893536,
                    4.96,
                    4.281768,
                    3.7667074,
                    3.362256,
                    3.0362391,
                    2.7678573,
                    2.5430682,
                    2.3520486,
                    2.1877205,
                    2.0448549,
                    1.9195048,
                    1.808635,
                    1.7098732,
                    1.6213388,
                    1.5415217,
                    1.4691944,
                    1.40335,
                    1.3431542,
                    1.2879103,
                    1.2370312,
                    1.1900192,
                    1.1464497,
                    1.105958,
                    1.0682288,
                    1.032989,
                    1 };

        for (int i = 0; i < 32; i++) 
        {
            //Find the Standard Shader
            Material myNewMaterial = new Material(Shader.Find("Skybox/PanoramicBeta"));
            //Set Texture on the material
            
            string filename = folder + "/courtyard_" + i.ToString();
            myNewMaterial.SetTexture("_Tex", Resources.Load(filename) as Texture);
            // Apply to GameObject
            // trans.GetComponent<MeshRenderer>().material = myNewMaterial;

            GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            Renderer renderer = cylinder.GetComponent<Renderer>();
            renderer.material = myNewMaterial;
            
            cylinder.transform.position = new Vector3(0, 0, 0);
            cylinder.transform.localScale = new Vector3((float)depths[i] * 6, (float)depths[i] * 6, (float)depths[i] * 6);

            Mesh mesh = cylinder.GetComponent<MeshFilter>().mesh;
        }
        //Vector3[] normals = mesh.normals;

        //for (int i = 0; i < normals.Length; i++)
        //{
        //    normals[i] = -normals[i];
        //}
        //mesh.normals = normals;
        
        /*Vector3[] vertices = mesh.vertices;
        Vector2[] uv = mesh.uv;
        Vector3[] normals = mesh.normals;
        int szV = vertices.Length;
        Vector3[] newVerts = new Vector3[szV * 2];
        Vector2[] newUv = new Vector2[szV * 2];
        Vector3[] newNorms = new Vector3[szV * 2];
        for (int j = 0; j < szV; j++)
        {
            // duplicate vertices and uvs:
            newVerts[j] = newVerts[j + szV] = vertices[j];
            newUv[j] = newUv[j + szV] = uv[j];
            // copy the original normals...
            newNorms[j] = normals[j];
            // and revert the new ones
            newNorms[j + szV] = -normals[j];
        }
        int[] triangles = mesh.triangles;
        int szT = triangles.Length;
        int[] newTris = new int[szT * 2]; // double the triangles
        for (int i = 0; i < szT; i += 3)
        {
            // copy the original triangle
            newTris[i] = triangles[i];
            newTris[i + 1] = triangles[i + 1];
            newTris[i + 2] = triangles[i + 2];
            // save the new reversed triangle
            int j = i + szT;
            newTris[j] = triangles[i] + szV;
            newTris[j + 2] = triangles[i + 1] + szV;
            newTris[j + 1] = triangles[i + 2] + szV;
        }*/
        //mesh.vertices = newVerts;
        //mesh.uv = newUv;
        //mesh.normals = newNorms;
        //mesh.triangles = newTris; // assign triangles last!

    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
