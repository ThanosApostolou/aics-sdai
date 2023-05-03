using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class FindPathAStar
{
    public FindPath findPath;

    public FindPathAStar(FindPath findPath)
    {
        this.findPath = findPath;
    }
    

    public void Search()
    {
        Debug.Log("Start AStar Search()");
        PathMarker thisNode = this.findPath.lastPos;

        if (thisNode.Equals(findPath.goalNode))
        {
            findPath.done = true;
            // Debug.Log("DONE!");
            return;
        }

        foreach (MapLocation dir in findPath.maze.directions)
        {
            MapLocation neighbour = dir + thisNode.location;

            if (findPath.maze.map[neighbour.x, neighbour.z] == 1) continue;
            if (neighbour.x < 1 || neighbour.x >= findPath.maze.width || neighbour.z < 1 || neighbour.z >= findPath.maze.depth) continue;
            if (IsClosed(neighbour)) continue;

            float g = Vector2.Distance(thisNode.location.ToVector(), neighbour.ToVector()) + thisNode.G;
            float h = Vector2.Distance(neighbour.ToVector(), findPath.goalNode.location.ToVector());
            float f = g + h;

            GameObject pathBlock = FindPath.Instantiate(findPath.pathP,
                new Vector3(neighbour.x * findPath.maze.scale, 0.0f, neighbour.z * findPath.maze.scale), Quaternion.identity);

            TextMesh[] values = pathBlock.GetComponentsInChildren<TextMesh>();

            values[0].text = "G: " + g.ToString("0.00");
            values[1].text = "H: " + h.ToString("0.00");
            values[2].text = "F: " + f.ToString("0.00");

            if (!UpdateMarker(neighbour, g, h, f, thisNode))
            {
                findPath.open.Add(new PathMarker(neighbour, g, h, f, pathBlock, thisNode));
            }
        }

        findPath.open = findPath.open.OrderBy(p => p.F).ThenBy(n => n.H).ToList<PathMarker>();
        PathMarker pm = (PathMarker)findPath.open.ElementAt(0);
        findPath.closed.Add(pm);

        findPath.open.RemoveAt(0);
        pm.marker.GetComponent<Renderer>().material = findPath.closedMaterial;

        findPath.lastPos = pm;
    }

    bool UpdateMarker(MapLocation pos, float g, float h, float f, PathMarker prt)
    {
        foreach (PathMarker p in findPath.open)
        {
            if (p.location.Equals(pos))
            {
                p.G = g;
                p.H = h;
                p.F = f;
                p.parent = prt;
                return true;
            }
        }

        return false;
    }

    bool IsClosed(MapLocation marker)
    {
        foreach (PathMarker p in findPath.closed)
        {
            if (p.location.Equals(marker)) return true;
        }

        return false;
    }
    
}