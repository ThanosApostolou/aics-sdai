import "../../App.css";
import Select from 'react-select';
import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";

function AddAdmin(props) {

  const [Roles, setRoles] = useState([]);
  const [Users, setUsers] = useState([]);
  const [selectedRole, setSelectedRole] = useState("");
  const [selectedUser, setSelectedUser] = useState("");
  const [RoleId, setRoleId] = useState("");
  const [UserId, setUserId] = useState("");
  const [Description, setDescription] = useState("");

  const add = (e) => {
    e.preventDefault();
    if (RoleId === "" || UserId === "" || Description === "") {
      alert("ALL the fields are mandatory!");
      return;
    }
    const adminToAdd = {
      RoleId: RoleId, UserId: UserId, Description: Description,
      Role: { Id: RoleId, Name: "", Description: "" },
      User: { Id: UserId, Username: "", Email: "", Address: "" }
    };
    props.addAdminHandler(adminToAdd);
    setRoleId("");
    setUserId("");
    setDescription("");
  };

  const handleChangeRole = (selectedRole) => {
    setSelectedRole(selectedRole);
    setRoleId(selectedRole.value);
  };

  const handleChangeUser = (selectedUser) => {
    setSelectedUser(selectedUser);
    setUserId(selectedUser.value);
  };

  const retrieveRoles = async () => {
    const response = await api.get("/role");
    return response.data;
  }

  const retrieveUsers = async () => {
    const response = await api.get("/eshopUser");
    return response.data;
  }

  useEffect(() => {
    const getAllRoles = async () => {
      const allRoles = await retrieveRoles();
      if (allRoles) setRoles(
        allRoles.map((role) => {
          return {
            label: role.Name,
            value: role.Id
          }
        })
      );
    };

    const getAllUsers = async () => {
      const allUsers = await retrieveUsers();
      if (allUsers) setUsers(
        allUsers.map((user) => {
          return {
            label: user.Username,
            value: user.Id
          }
        })
      );
    };

    getAllUsers();
    getAllRoles();
  }, []);

  return (
    <div className="ui main">
      <h2>Add Admin</h2>
      <br></br>
      <form className="ui form" onSubmit={add}>
        <div className="field">
          <label>User</label>
          <Select
            value={selectedUser}
            onChange={handleChangeUser}
            options={Users}
          />
        </div>
        <div className="field">
          <label>Role</label>
          <Select
            value={selectedRole}
            onChange={handleChangeRole}
            options={Roles}
          />
        </div>
        <div className="field">
          <label>
            Description:
            <input
              type="text"
              name="description"
              placeholder="Description"
              value={Description}
              onChange={e => setDescription(e.target.value)}
            />
          </label>
        </div>
        <button className="buttonInsert">Add</button>
      </form>
    </div>
  );

}

export default AddAdmin;
