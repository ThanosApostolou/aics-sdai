import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import "../../App.css";
import AddAdminModal from "./AddAdminModal";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";
import Select from 'react-select';

function AdminDataDisplay() {
    const [Roles, setRoles] = useState([]);
    const [selectedRole, setSelectedRole] = useState("");
    const [Users, setUsers] = useState([]);
    const [selectedUser, setSelectedUser] = useState("");
    const [admins, setAdmins] = useState([]);
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

    const handleChangeRole = (selectedRole) => {
        setSelectedRole(selectedRole);
    };

    const handleChangeUser = (selectedUser) => {
        setSelectedUser(selectedUser);
    };

    const retrieveAdmins = async () => {
        const response = await api.get("/admin");
        return response.data;
    }

    const retrieveRoles = async () => {
        const response = await api.get("/role");
        return response.data;
    }

    const retrieveUsers = async () => {
        const response = await api.get("/eshopUser");
        return response.data;
    }

    async function addAdminHandler(admin) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = { ...admin }
                const response = await api.post("/admin", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            toast.error("Failed to Add!")
        }
    };

    const removeAdminHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/admin/${id}`);
                const newAdminList = admins.filter((admin) => {
                    return admin.AdminID !== id;
                });
                setAdmins(newAdminList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!")
        }
    };

    const updateAdminHandler = async (admin) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const adminToUpdate = {
                    Id: admin.Id,
                    RoleId: selectedRole ? selectedRole.value.Id : admin.RoleId,
                    UserId: selectedUser ? selectedUser.value.Id : admin.UserId,
                    Description: admin.Description,
                    Role: selectedRole ?
                        { Id: selectedRole.value.Id, Name: selectedRole.value.Name, Description: selectedRole.value.Description }
                        :
                        { Id: admin.admin, Name: "", Description: "" },
                    User: selectedUser ?
                        { Id: selectedUser.value.Id, Username: selectedUser.value.Username, Email: selectedUser.value.Email, Address: selectedUser.value.Address}
                        :
                        { Id: admin.UserId, Username: "", Email: "", Address: "" }
                };
                console.log(adminToUpdate);
                await api.put("/admin", adminToUpdate);
                setAdmins(
                    admins.map((existingAdmin) => {
                        return existingAdmin.RoleId === adminToUpdate.RoleId
                            ? { ...adminToUpdate }
                            : existingAdmin;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            toast.error("Failed to update!");
        }
    };

    const onDescriptionUpdate = (shop, event) => {
        const { value } = event.target;
        const data = [...rows];
        shop.Description = value;
        initRow(data);
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllAdmins = async () => {
            const allAdmins = await retrieveAdmins();
            if (allAdmins) setAdmins(allAdmins);
        };

        const getAllRokes = async () => {
            const allRoles = await retrieveRoles();

            if (allRoles) setRoles(
                allRoles.map((Role) => {
                    return {
                        label: Role.Name,
                        value: Role
                    }
                })
            );

        };

        const getAllUsers = async () => {
            const allUsers = await retrieveUsers();

            if (allUsers) setUsers(
                allUsers.map((User) => {
                    return {
                        label: User.Username,
                        value: User
                    }
                })
            );
        };

        getAllAdmins();
        getAllRokes();
        getAllUsers();
        setDeleted(false);
        setUpdated(false);
        setAdded(false);


    }, [added, deleted, updated]);

    const DisplayData = admins.map(
        (admin) => {
            return (
                <tr key={admin.Id}>
                    <td>
                        {admin.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={admin.Role.Name}
                            name="adminRoleID"
                            className="form-control"
                        />
                        <Select
                            value={selectedRole}
                            onChange={handleChangeRole}
                            options={Roles}
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={admin.User.Username}
                            name="AdminUserId"
                            className="form-control"
                        />
                        <Select
                            value={selectedUser}
                            onChange={handleChangeUser}
                            options={Users}
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={admin.Description}
                            onChange={(event) => onDescriptionUpdate(admin, event)}
                            name="description"
                            className="form-control"
                        />
                    </td>

                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updateAdminHandler(admin)}
                        >
                            Update
                        </button>
                        <span> </span>
                        <button
                            className="buttonDelete"
                            onClick={() => removeAdminHandler(admin.Id)}
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
            <AddAdminModal addAdminHandler={addAdminHandler} open={open} handleOpen={handleOpen} />
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Role</th>
                        <th>User</th>
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

export default AdminDataDisplay;