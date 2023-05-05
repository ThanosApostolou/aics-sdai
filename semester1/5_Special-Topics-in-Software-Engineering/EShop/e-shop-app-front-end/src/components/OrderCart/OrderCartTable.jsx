import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import "../../App.css";
import AddOrderCartModal from "./AddOrderCartModal";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";
import Select from 'react-select';

function OrderCartDataDisplay() {
    const [Carts, setCarts] = useState([]);
    const [selectedCart, setSelectedCart] = useState("");
    const [Customers, setCustomers] = useState([]);
    const [selectedCustomer, setSelectedCustomer] = useState("");
    const [Payments, setPayments] = useState([]);
    const [selectedPayment, setSelectedPayment] = useState("");
    const [orderCarts, setOrderCarts] = useState([]);
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

    const handleChangeCart = (selectedCart) => {
        setSelectedCart(selectedCart);
    };

    const handleChangeCustomer = (selectedCustomer) => {
        setSelectedCustomer(selectedCustomer);
    };

    const handleChangePayment = (selectedPayment) => {
        setSelectedPayment(selectedPayment);
    };

    const retrieveOrderCarts = async () => {
        const response = await api.get("/orderCart");
        return response.data;
    }

    const retrieveCarts = async () => {
        const response = await api.get("/cart");
        return response.data;
    }

    const retrieveCustomers = async () => {
        const response = await api.get("/eshopUser");
        return response.data;
    }

    const retrievePayments = async () => {
        const response = await api.get("/payment");
        return response.data;
    }

    async function addOrderCartHandler(orderCart) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = { ...orderCart }
                const response = await api.post("/orderCart", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            console.log(orderCart)
            toast.error("Failed to Add!")
        }
    };

    const removeOrderCartHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/orderCart/${id}`);
                const newOrderCartList = orderCarts.filter((orderCart) => {
                    return orderCart.OrderCartID !== id;
                });
                setOrderCarts(newOrderCartList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!")
        }
    };

    const updateOrderCartHandler = async (orderCart) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const orderCartToUpdate = {
                    Id: orderCart.Id,
                    DeliveryAdress :orderCart.DeliveryAdress,
                    Date :orderCart.Date,
                    Cart: selectedCart ? selectedCart.value.Id : orderCart.Cart,
                    Customer: selectedCustomer ? selectedCustomer.value.Id : orderCart.Customer,
                    Payment: selectedPayment ? selectedPayment.value.Id : orderCart.Payment,
                    CartNavigation: selectedCart ?
                        { Id: selectedCart.value.Id, Quantity: selectedCart.value.Quantity, Customer: selectedCart.value.Customer,
                            CustomerNavigation: {Id: selectedCart.value.Customer, Username: "", Email: "", Address: ""} }
                        :
                        { Id: orderCart.value.Cart, Quantity: orderCart.value.Quantity, Customer: orderCart.value.Customer,
                            CustomerNavigation: {Id: orderCart.value.Customer, Username: "", Email: "", Address: ""} },
                        CustomerNavigation: selectedCustomer ?
                        { Id: selectedCustomer.value.Id, Username: selectedCustomer.value.Username, Email: selectedCustomer.value.Email, Address:selectedCustomer.value.Address }
                        :
                        { Id: orderCart.Customer, Username: "", Email: "", Address:"" },
                        PaymentNavigation: selectedPayment ?
                        { Id: selectedPayment.value.Id, Availability: selectedPayment.value.Availability, Amount: selectedPayment.value.Amount, PaymentCategoryId: selectedPayment.value.PaymentCategoryId,
                            PaymentCategory: { Id: selectedPayment.value.Id, Name: "", Description: "" }}
                        :
                        { Id: orderCart.Payment, Availability: false, Amount: "", PaymentCategoryId: "",
                            PaymentCategory: { Id: orderCart.Payment, Name: "", Description: "" }}
                };
                console.log(orderCartToUpdate);
                await api.put("/orderCart", orderCartToUpdate);
                setOrderCarts(
                    orderCarts.map((existingOrderCart) => {
                        return existingOrderCart.Id === orderCartToUpdate.Id
                            ? { ...orderCartToUpdate }
                            : existingOrderCart;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            console.log(e);
            toast.error("Failed to update!");
        }
    };

    const onDeliveryAdressUpdate = (orderCart, event) => {
        const { value } = event.target;
        const data = [...rows];
        orderCart.DeliveryAdress = value;
        initRow(data);
    };

    const onDateUpdate = (orderCart, event) => {
        const { value } = event.target;
        const data = [...rows];
        orderCart.Date = value;
        initRow(data);
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllOrderCarts = async () => {
            const allOrderCarts = await retrieveOrderCarts();
            if (allOrderCarts) setOrderCarts(allOrderCarts);
        };

        const getAllCarts = async () => {
            const allCarts = await retrieveCarts();

            if (allCarts) setCarts(
                allCarts.map((Cart) => {
                    return {
                        label: Cart.Id,
                        value: Cart
                    }
                })
            );

        };

        const getAllCustomers = async () => {
            const allCustomers = await retrieveCustomers();

            if (allCustomers) setCustomers(
                allCustomers.map((Customer) => {
                    return {
                        label: Customer.Username,
                        value: Customer
                    }
                })
            );
        };

        const getAllPayments = async () => {
            const allPayments = await retrievePayments();

            if (allPayments) setPayments(
                allPayments.map((Payment) => {
                    return {
                        label: Payment.Id,
                        value: Payment
                    }
                })
            );
        };

        getAllOrderCarts();
        getAllCarts();
        getAllCustomers();
        getAllPayments();
        setDeleted(false);
        setUpdated(false);
        setAdded(false);


    }, [added, deleted, updated]);

    const DisplayData = orderCarts.map(
        (orderCart) => {
            return (
                <tr key={orderCart.Id}>
                    <td>
                        {orderCart.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={orderCart.CartNavigation.Id}
                            name="cartProductID"
                            className="form-control"
                        />
                        <Select
                            value={selectedCart}
                            onChange={handleChangeCart}
                            options={Carts}
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={orderCart.CustomerNavigation.Username}
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
                            disabled={true}
                            value={orderCart.Payment}
                            name="PaymentId"
                            className="form-control"
                        />
                        <Select
                            value={selectedPayment}
                            onChange={handleChangePayment}
                            options={Payments}
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={orderCart.Date}
                            onChange={(event) => onDateUpdate(orderCart, event)}
                            name="Date"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={orderCart.DeliveryAdress}
                            onChange={(event) => onDeliveryAdressUpdate(orderCart, event)}
                            name="DeliveryAdress"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updateOrderCartHandler(orderCart)}
                        >
                            Update
                        </button>
                        <span> </span>
                        <button
                            className="buttonDelete"
                            onClick={() => removeOrderCartHandler(orderCart.Id)}
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
            <AddOrderCartModal addOrderCartHandler={addOrderCartHandler} open={open} handleOpen={handleOpen} />
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Cart</th>
                        <th>Customer</th>
                        <th>Payment</th>
                        <th>Date</th>
                        <th>Delivery Address</th>
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

export default OrderCartDataDisplay;