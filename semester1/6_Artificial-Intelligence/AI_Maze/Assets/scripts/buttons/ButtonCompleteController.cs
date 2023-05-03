using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;

public class ButtonCompleteController : MonoBehaviour
{
    private Button _buttonCompleteGameObject;
    public GameObject mazeGameObject;

    private FindPath _findPath;
    
    // Start is called before the first frame update
    void Start()
    {
        this._buttonCompleteGameObject = GetComponent<Button>();
        // Transform.Find("Maze");
        this._findPath = mazeGameObject.GetComponent<FindPath>();
        this._buttonCompleteGameObject.onClick.AddListener(this.OnClick);

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void OnClick()
    {
        this._findPath.Complete();
    }
}
