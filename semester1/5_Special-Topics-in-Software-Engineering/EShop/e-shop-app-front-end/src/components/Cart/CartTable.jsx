import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import "../../App.css";
import AddCartModal from "./AddCartModal";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";
import Select from 'react-select';

function CartDataDisplay() {
    const [Customers, setCustomers] = useState([]);
    const [selectedCustomer, setSelectedCustomer] = useState("");
    const [carts, setCarts] = useState([]);
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

    const handleChangeCustomer = (selectedCustomer) => {
        setSelectedCustomer(selectedCustomer);
    };

    const retrieveCarts = async () => {
        const response = await api.get("/cart");
        return response.data;
    }

    const retrieveCustomers = async () => {
        const response = await api.get("/eshopUser");
        return response.data;
    }

    async function addCartHandler(cart) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = { ...cart }
                const response = await api.post("/cart", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            toast.error("Failed to Add!")
        }
    };

    const removeCartHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/cart/${id}`);
                const newCartList = carts.filter((cart) => {
                    return cart.CartID !== id;
                });
                setCarts(newCartList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!")
        }
    };

    const updateCartHandler = async (cart) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const cartToUpdate = {
                    Id: cart.Id,
                    Customer: selectedCustomer ? selectedCustomer.value.Id : cart.UserId,
                    Quantity: cart.Quantity,
                    CustomerNavigation: selectedCustomer ?
                        { Id: selectedCustomer.value.Id, Username: selectedCustomer.value.Username, Email: selectedCustomer.value.Email, Address: selectedCustomer.value.Address}
                        :
                        { Id: cart.UserId, Username: "", Email: "", Address: "" }
                };
                console.log(cartToUpdate);
                await api.put("/cart", cartToUpdate);
                setCarts(
                    carts.map((existingCart) => {
                        return existingCart.Id === cartToUpdate.RoleId
                            ? { ...cartToUpdate }
                            : existingCart;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            toast.error("Failed to update!");
        }
    };

    const onQuantityUpdate = (cart, event) => {
        const { value } = event.target;
        const data = [...rows];
        cart.Quantity = value;
        initRow(data);
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllCarts = async () => {
            const allCarts = await retrieveCarts();
            if (allCarts) setCarts(allCarts);
        };

        const getAllCustomers = async () => {
            const allCustomers = await retrieveCustomers();

            if (allCustomers) setCustomers(
                allCustomers.map((customer) => {
                    return {
                        label: customer.Username,
                        value: customer
                    }
                })
            );
        };

        getAllCarts();
        getAllCustomers();
        setDeleted(false);
        setUpdated(false);
        setAdded(false);


    }, [added, deleted, updated]);

    const DisplayData = carts.map(
        (cart) => {
            return (
                <tr key={cart.Id}>
                    <td>
                        {cart.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={cart.CustomerNavigation.Username}
                            name="CustomerNavigationId"
                            className="form-control"
                        />
                        <Select
                            value={selectedCustomer}
                            onChange={handleChangeCustomer}
                            options={Customers}
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={cart.Quantity}
                            onChange={(event) => onQuantityUpdate(cart, event)}
                            name="quantity"
                            className="form-control"
                        />
                    </td>

                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updateCartHandler(cart)}
                        >
                            Update
                        </button>
                        <span> </span>
                        <button
                            className="buttonDelete"
                            onClick={() => removeCartHandler(cart.Id)}
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
            <AddCartModal addCartHandler={addCartHandler} open={open} handleOpen={handleOpen} />
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>User</th>
                        <th>Quantity</th>
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

export default CartDataDisplay;