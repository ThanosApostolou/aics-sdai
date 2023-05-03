using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public enum AlgorithmEnum
{
    AStar,
    Dijkstra,
    FloodFill,
}

public class PathMarker
{
    public MapLocation location;
    public float G, H, F;
    public GameObject marker;
    public PathMarker parent;

    public PathMarker(MapLocation l, float g, float h, float f, GameObject m, PathMarker p)
    {
        location = l;
        G = g;
        H = h;
        F = f;
        marker = m;
        parent = p;
    }

    public override bool Equals(object obj)
    {
        if ((obj == null) || !this.GetType().Equals(obj.GetType()))
            return false;
        else
            return location.Equals(((PathMarker)obj).location);
    }

    public override int GetHashCode()
    {
        return 0;
    }
}

public class FindPath : MonoBehaviour
{
    public Maze maze;
    public Material closedMaterial;
    public Material openMaterial;
    public GameObject start;
    public GameObject end;
    public GameObject pathP;

    public PathMarker startNode;
    public PathMarker goalNode;
    public PathMarker lastPos;
    public bool done = false;
    public bool hasStarted = false;

    public GameObject player;

    public List<PathMarker> open = new List<PathMarker>();
    public List<PathMarker> closed = new List<PathMarker>();

    public List<Vector3> pathPoints = new List<Vector3>();
    private int currentPointIndex = 0;

    private FindPathAStar _findPathAStar;
    private FindPathDijkstra _findPathDijkstra;
    private FindPathFloodFill _findPathFloodFill;
    private AlgorithmEnum _algorithmEnum = AlgorithmEnum.AStar;

    void RemoveAllMarkers(bool removeEnd)
    {
        GameObject[] markers = GameObject.FindGameObjectsWithTag("marker");
        foreach (GameObject m in markers)
        {
            if (m is not null)
            {
                Destroy(m);
            }
        }

        if (removeEnd)
        {
            GameObject endMarker = GameObject.FindGameObjectWithTag("endMarker");
            if (endMarker is not null)
            {
                Destroy(endMarker);
            }
        }
    }

    public void BeginSearch()
    {
        hasStarted = true;
        pathPoints.Clear();

        done = false;
        RemoveAllMarkers(true);

        GameObject playerToDestroy = GameObject.FindGameObjectWithTag("Player");
        Destroy(playerToDestroy);

        List<MapLocation> locations = new List<MapLocation>();

        for (int z = 1; z < maze.depth - 1; ++z)
        {
            for (int x = 1; x < maze.width - 1; ++x)
            {
                if (maze.map[x, z] != 1)
                {
                    locations.Add(new MapLocation(x, z));
                }
            }
        }

        locations.Shuffle();

        Vector3 startLocation = new Vector3(locations[0].x * maze.scale, 0.0f, locations[0].z * maze.scale);
        // startNode = new PathMarker(new MapLocation(locations[0].x, locations[0].z),
        //     0.0f, 0.0f, 0.0f, Instantiate(start, startLocation, Quaternion.identity), null);

        startNode = new PathMarker(new MapLocation(locations[0].x, locations[0].z),
            0.0f, 0.0f, 0.0f, Instantiate(start, startLocation, Quaternion.identity), null);
        // Get the renderer component of the object
        Renderer renderer = startNode.marker.GetComponent<Renderer>();

        // Disable the renderer to make the object invisible
        renderer.enabled = false;

        // Create the player game object and set its position
        GameObject playerObject = Instantiate(player.gameObject, startLocation, Quaternion.identity);

        // Assign the player game object to the playerTransform variable
        //player = playerObject.transform;

        Vector3 endLocation = new Vector3(locations[1].x * maze.scale, 0.0f, locations[1].z * maze.scale);
        goalNode = new PathMarker(new MapLocation(locations[1].x, locations[1].z),
            0.0f, 0.0f, 0.0f, Instantiate(end, endLocation, Quaternion.identity), null);

        open.Clear();
        closed.Clear();

        open.Add(startNode);
        lastPos = startNode;
    }

    public void Search()
    {
        PathMarker thisNode = lastPos;
        if (!hasStarted || done)
        {
            return;
        }

        switch (_algorithmEnum)
        {
            case AlgorithmEnum.AStar:
            {
                this._findPathAStar.Search();
                break;
            }
            case AlgorithmEnum.Dijkstra:
            {
                this._findPathDijkstra.Search();
                break;
            }
            case AlgorithmEnum.FloodFill:
            {
                this._findPathFloodFill.Search();
                break;
            }
        }
    }

    void Start()
    {
        this._findPathAStar = new FindPathAStar(this);
        this._findPathDijkstra = new FindPathDijkstra(this);
        this._findPathFloodFill = new FindPathFloodFill(this);
    }

    private void GetPath()
    {
        if (!hasStarted)
        {
            return;
        }

        RemoveAllMarkers(false);
        PathMarker begin = lastPos;

        while (!startNode.Equals(begin) && begin != null)
        {
            pathPoints.Add(new Vector3(begin.location.x * maze.scale, 0, begin.location.z * maze.scale));

            //Instantiate(pathP, new Vector3(begin.location.x * maze.scale, 0, begin.location.z * maze.scale), 
            //Quaternion.identity);


            begin = begin.parent;
        }

        Vector3 startNodePoint = new Vector3(startNode.location.x * maze.scale, 0, startNode.location.z * maze.scale);
        pathPoints.Add(startNodePoint);


        ReverseList(pathPoints);
        // Instantiate(pathP, new Vector3(startNode.location.x * maze.scale, 0, startNode.location.z * maze.scale), 
        //Quaternion.identity);
    }

    public float moveSpeed = 2f; // speed at which the player moves along the path
    public float lerpSpeed = 0.1f; // speed at which the player moves between path points
    public int segmentsPerUnit = 1000; // number of interpolation segments per unit of distance

    public void MovePlayer()
    {
        if (!hasStarted || !done)
        {
            return;
        }

        GetPath();
        StartCoroutine(MovePlayerCoroutine());
    }

    IEnumerator MovePlayerCoroutine()
    {
        player = GameObject.FindGameObjectWithTag("Player");
        int currentPointIndex = 0;
        float distanceToNextPoint = 0f;

        while (currentPointIndex < pathPoints.Count - 1)
        {
            Vector3 currentPoint = pathPoints[currentPointIndex];
            Vector3 nextPoint = pathPoints[currentPointIndex + 1];

            distanceToNextPoint = Vector3.Distance(currentPoint, nextPoint);
            int numSegments = Mathf.CeilToInt(distanceToNextPoint * segmentsPerUnit);

            for (int i = 0; i < numSegments; i++)
            {
                // calculate the percentage of the distance between current and next points
                // that the player should move based on the lerpSpeed
                float distanceMoved = (moveSpeed / segmentsPerUnit) * Time.deltaTime;
                float percentageMoved = Mathf.Clamp01(distanceMoved / distanceToNextPoint);

                // interpolate the player's position between current and next points
                Vector3 segmentStart = Vector3.Lerp(currentPoint, nextPoint, i / (float)numSegments);
                Vector3 segmentEnd = Vector3.Lerp(currentPoint, nextPoint, (i + 1) / (float)numSegments);
                player.transform.position = Vector3.Lerp(segmentStart, segmentEnd, moveSpeed * Time.deltaTime);

                // wait for the next frame
                yield return null;
            }

            // move to the next point in the path
            currentPointIndex++;
        }
    }


    public void ReverseList(List<Vector3> myList)
    {
        myList.Reverse();
    }


    public void Complete()
    {
        if (!hasStarted || done)
        {
            return;
        }

        while (!done)
        {
            Search();
        }
    }

    public void SetAStar()
    {
        this._algorithmEnum = AlgorithmEnum.AStar;
        this.BeginSearch();
    }
    
    public void SetDijkstra()
    {
        this._algorithmEnum = AlgorithmEnum.Dijkstra;
        this.BeginSearch();
    }
    public void SetFloodFill()
    {
        this._algorithmEnum = AlgorithmEnum.FloodFill;
        this.BeginSearch();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.B))
        {
            BeginSearch();
        }
        if (Input.GetKeyDown(KeyCode.S))
        {
            Search();
        }
        if (Input.GetKeyDown(KeyCode.C))
        {
            Complete();
        }
        if (Input.GetKeyDown(KeyCode.M))
        {
            MovePlayer();
        }
    }
}