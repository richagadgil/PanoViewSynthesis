using System.Collections;
using System.Collections.Generic;
using UnityEngine;

 
public class CamRotation: MonoBehaviour {
 
    public float rotationSpeed = 1.0f;
    

    void FixedUpdate() {
        Vector3 rotation = transform.eulerAngles;
 
        rotation.x += Input.GetAxis("Horizontal") * rotationSpeed * Time.deltaTime; // Standart Left-/Right Arrows and A & D Keys
 
        transform.eulerAngles = rotation;
    }
}