import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import AddPaymentCategoryModal from "./AddPaymentCategoryModal";
import "../../App.css";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";

function PaymentCategoryDataDisplay(props) {
    const [paymentCategories, setPaymentCategories] = useState([]);
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

    const retrievePaymentCategories = async () => {
        const response = await api.get("/paymentCategory");
        return response.data;
    }

    async function addPaymentCategoryHandler(paymentCategory) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = {
                    ...paymentCategory
                }

                const response = await api.post("/paymentCategory", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            toast.error("Failed to Add!")
        }
    };

    const removePaymentCategoryHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/paymentCategory/${id}`);
                const newPaymentCategoryList = paymentCategories.filter((paymentCategory) => {
                    return paymentCategory.PaymentCategoryID !== id;
                });
                setPaymentCategories(newPaymentCategoryList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!");
        }
    };

    const updatePaymentCategoryHandler = async (paymentCategory) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const paymentCategoryToUpdate = {
                    Id: paymentCategory.Id,
                    Name: paymentCategory.Name,
                    Description: paymentCategory.Description
                };
                console.log(paymentCategoryToUpdate);
                const response = await api.put("/paymentCategory", paymentCategory);
                const { paymentCategoryName } = response.data;
                setPaymentCategories(
                    paymentCategories.map((paymentCategory) => {
                        return paymentCategory.Name === paymentCategoryName ? { ...response.data } : paymentCategory;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            toast.error("Failed to update!");
        }
    };

    const onPaymentCategoryNameUpdate = (paymentCategory, event) => {
        const { value } = event.target;
        const data = [...rows];
        paymentCategory.Name = value;
        initRow(data);
        console.log(paymentCategory)
    };

    const onDescriptionUpdate = (paymentCategory, event) => {
        const { value } = event.target;
        const data = [...rows];
        paymentCategory.Description = value;
        initRow(data);
        console.log(paymentCategory)
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllPaymentCategories = async () => {
            const allPaymentCategories = await retrievePaymentCategories();
            if (allPaymentCategories) setPaymentCategories(allPaymentCategories);
        };

        getAllPaymentCategories();
        setAdded(false);
        setDeleted(false);
        setUpdated(false);

    }, [added, deleted, updated]);

    const DisplayData = paymentCategories.map(
        (paymentCategory) => {
            return (
                <tr key={paymentCategory.Id}>
                    <td>
                        {paymentCategory.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            value={paymentCategory.Name}
                            onChange={(event) => onPaymentCategoryNameUpdate(paymentCategory, event)}
                            name="name"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={paymentCategory.Description}
                            onChange={(event) => onDescriptionUpdate(paymentCategory, event)}
                            name="description"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updatePaymentCategoryHandler(paymentCategory)}
                        >
                            Update
                        </button>
                        <button
                            className="buttonDelete"
                            onClick={() => removePaymentCategoryHandler(paymentCategory.Id)}
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
            <AddPaymentCategoryModal addPaymentCategoryHandler={addPaymentCategoryHandler} open={open} handleOpen={handleOpen} />
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

export default PaymentCategoryDataDisplay;