using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;

public class ButtonMoveController : MonoBehaviour
{
    private Button _assignedGameObject;
    public GameObject mazeGameObject;

    private FindPath _findPath;
    
    // Start is called before the first frame update
    void Start()
    {
        this._assignedGameObject = GetComponent<Button>();
        // Transform.Find("Maze");
        this._findPath = mazeGameObject.GetComponent<FindPath>();
        this._assignedGameObject.onClick.AddListener(this.OnClick);

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void OnClick()
    {
        this._findPath.MovePlayer();
    }
}
