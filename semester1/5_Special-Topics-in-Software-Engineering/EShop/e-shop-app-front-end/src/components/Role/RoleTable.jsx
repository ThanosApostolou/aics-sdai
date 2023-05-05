import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import AddRoleModal from "./AddRoleModal";
import "../../App.css";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";

function RoleDataDisplay(props) {
    const [roles, setRoles] = useState([]);
    const [added, setAdded] = useState(false);
    const [deleted, setDeleted] = useState(false);
    const [updated, setUpdated] = useState(false);
    const [open, setOpen] = useState(false);
    const options = {
        labels: {
            confirmable: "Confirm",
            cancellable: "Cancel"
        }
    }

    function handleOpen() {
        setOpen(!open);
    }

    const retrieveRoles = async () => {
        const response = await api.get("/role");
        return response.data;
    }

    async function addRoleHandler(role) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = {
                    ...role
                }

                const response = await api.post("/role", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            toast.error("Failed to Add!")
        }
    };

    const removeRoleHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/role/${id}`);
                const newRoleList = roles.filter((role) => {
                    return role.RoleID !== id;
                });
                setRoles(newRoleList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!");
        }
    };

    const updateRoleHandler = async (role) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const roleToUpdate = {
                    Id: role.Id,
                    Name: role.Name,
                    Description: role.Description
                };
                console.log(roleToUpdate);
                const response = await api.put("/role", role);
                const { roleName } = response.data;
                setRoles(
                    roles.map((role) => {
                        return role.Name === roleName ? { ...response.data } : role;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            toast.error("Failed to update!");
        }
    };

    const onRoleNameUpdate = (role, event) => {
        const { value } = event.target;
        const data = [...rows];
        role.Name = value;
        initRow(data);
        console.log(role)
    };

    const onDescriptionUpdate = (role, event) => {
        const { value } = event.target;
        const data = [...rows];
        role.Description = value;
        initRow(data);
        console.log(role)
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllRoles = async () => {
            const allRoles = await retrieveRoles();
            if (allRoles) setRoles(allRoles);
        };

        getAllRoles();
        setAdded(false);
        setDeleted(false);
        setUpdated(false);

    }, [added, deleted, updated]);

    const DisplayData = roles.map(
        (role) => {
            return (
                <tr key={role.Id}>
                    <td>
                        {role.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            value={role.Name}
                            onChange={(event) => onRoleNameUpdate(role, event)}
                            name="name"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={role.Description}
                            onChange={(event) => onDescriptionUpdate(role, event)}
                            name="description"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updateRoleHandler(role)}
                        >
                            Update
                        </button>
                        <button
                            className="buttonDelete"
                            onClick={() => removeRoleHandler(role.Id)}
                        >
                            Delete
                        </button>
                    </td>
                </tr>
            )
        }
    )
    return (
        <div>
            <AddRoleModal addRoleHandler={addRoleHandler} open={open} handleOpen={handleOpen} />
            <table>
                <thead>
                    <tr>
                        <th>Id</th>
                        <th>Name</th>
                        <th>Description</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {DisplayData}
                </tbody>
            </table>
        </div>
    )
}

export default RoleDataDisplay;