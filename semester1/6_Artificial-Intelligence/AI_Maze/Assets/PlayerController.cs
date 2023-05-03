using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class PlayerController : MonoBehaviour
{
    public float speed = 1f;
    public Transform characterTransform;

    private FindPath findPath;
    // Start is called before the first frame update
    void Start()
    {
        // Find the game object that has the other script attached
        GameObject pathScriptObject = GameObject.Find("Maze");

        // Get a reference to the other script
        FindPath findPath = pathScriptObject.GetComponent<FindPath>();

        characterTransform = findPath.player.transform;
    }

    // Update is called once per frame
    void Update()
    {
        if(findPath.pathPoints.Count > 0){
        Vector3 direction = (findPath.pathPoints[0] - characterTransform.position).normalized;
        characterTransform.position += direction * speed * Time.deltaTime;

        if(Vector3.Distance(characterTransform.position, findPath.pathPoints[0]) < 0.1f){
            findPath.pathPoints.RemoveAt(0);
        }
    }
    }
}
