import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import AddEshopUserModal from "./AddEshopUserModal";
import "../../App.css";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";

function EshopUserDataDisplay(props) {
    const [eshopUsers, setEshopUsers] = useState([]);
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

    const retrieveEshopUsers = async () => {
        const response = await api.get("/eshopUser");
        return response.data;
    }

    async function addEshopUserHandler(eshopUser) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = {
                    ...eshopUser
                }

                const response = await api.post("/eshopUser", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            toast.error("Failed to Add!")
        }
    };

    const removeEshopUserHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/eshopUser/${id}`);
                const newEshopUserList = eshopUsers.filter((eshopUser) => {
                    return eshopUser.EshopUserID !== id;
                });
                setEshopUsers(newEshopUserList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!");
        }
    };

    const updateEshopUserHandler = async (eshopUser) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const eshopUserToUpdate = {
                    Id: eshopUser.Id,
                    Username: eshopUser.Username,
                    Email: eshopUser.Email,
                    Address: eshopUser.Address,
                };
                const response = await api.put("/eshopUser", eshopUser);
                const { eshopUserName } = response.data;
                setEshopUsers(
                    eshopUsers.map((eshopUser) => {
                        return eshopUser.Username === eshopUserName ? { ...response.data } : eshopUser;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            toast.error("Failed to update!");
        }
    };

    const onEshopUserUsernameUpdate = (eshopUser, event) => {
        const { value } = event.target;
        const data = [...rows];
        eshopUser.Username = value;
        initRow(data);
        console.log(eshopUser)
    };

    const onEshopUserEmailUpdate = (eshopUser, event) => {
        const { value } = event.target;
        const data = [...rows];
        eshopUser.Email = value;
        initRow(data);
        console.log(eshopUser)
    };

    const onEshopUserAddressUpdate = (eshopUser, event) => {
        const { value } = event.target;
        const data = [...rows];
        eshopUser.Address = value;
        initRow(data);
        console.log(eshopUser)
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllEshopUsers = async () => {
            const allEshopUsers = await retrieveEshopUsers();
            if (allEshopUsers) setEshopUsers(allEshopUsers);
        };

        getAllEshopUsers();
        setAdded(false);
        setDeleted(false);
        setUpdated(false);

    }, [added, deleted, updated]);

    const DisplayData = eshopUsers.map(
        (eshopUser) => {
            return (
                <tr key={eshopUser.Id}>
                    <td>
                        {eshopUser.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            value={eshopUser.Username}
                            onChange={(event) => onEshopUserUsernameUpdate(eshopUser, event)}
                            name="username"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={eshopUser.Email}
                            onChange={(event) => onEshopUserEmailUpdate(eshopUser, event)}
                            name="email"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={eshopUser.Address}
                            onChange={(event) => onEshopUserAddressUpdate(eshopUser, event)}
                            name="address"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updateEshopUserHandler(eshopUser)}
                        >
                            Update
                        </button>
                        <button
                            className="buttonDelete"
                            onClick={() => removeEshopUserHandler(eshopUser.Id)}
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
            <AddEshopUserModal addEshopUserHandler={addEshopUserHandler} open={open} handleOpen={handleOpen} />
            <table>
                <thead>
                    <tr>
                        <th>Id</th>
                        <th>Username</th>
                        <th>Email</th>
                        <th>Address</th>
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

export default EshopUserDataDisplay;