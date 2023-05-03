using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;
using TMPro;
public class DropdownAlgorithmController : MonoBehaviour
{
    private TMP_Dropdown _dropdownAlgorithmGameObject;
    public GameObject mazeGameObject;

    private FindPath _findPath;
    
    // Start is called before the first frame update
    void Start()
    {
        this._dropdownAlgorithmGameObject = GetComponent<TMP_Dropdown>();
        this._findPath = mazeGameObject.GetComponent<FindPath>();
        this._dropdownAlgorithmGameObject.onValueChanged.AddListener(ValueChanged);

    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.A))
        {
            this._dropdownAlgorithmGameObject.value = 0;
        }
        if (Input.GetKeyDown(KeyCode.D))
        {
            this._dropdownAlgorithmGameObject.value = 1;
        }
        if (Input.GetKeyDown(KeyCode.F))
        {
            this._dropdownAlgorithmGameObject.value = 2;
        }
    }

    public void ValueChanged(int selection)
    {
        Debug.Log("selection: " + selection);
        AlgorithmEnum chosenAlgorithm = (AlgorithmEnum)selection;
        switch (chosenAlgorithm)
        {
            case AlgorithmEnum.AStar:
            {
                this._findPath.SetAStar();
                break;
            }
            case AlgorithmEnum.Dijkstra:
            {
                this._findPath.SetDijkstra();
                break;
            }
            case AlgorithmEnum.FloodFill:
            {
                this._findPath.SetFloodFill();
                break;
            }
        }
    }
}
