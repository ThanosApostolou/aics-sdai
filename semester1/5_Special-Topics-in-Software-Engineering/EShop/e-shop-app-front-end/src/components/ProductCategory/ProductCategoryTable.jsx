import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import AddProductCategoryModal from "./AddProductCategoryModal";
import "../../App.css";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";

function ProductCategoryDataDisplay(props) {
    const [productCategories, setProductCategories] = useState([]);
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

    const retrieveProductCategories = async () => {
        const response = await api.get("/productCategory");
        return response.data;
    }

    async function addProductCategoryHandler(productCategory) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = {
                    ...productCategory
                }

                const response = await api.post("/productCategory", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            toast.error("Failed to Add!")
        }
    };

    const removeProductCategoryHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/productCategory/${id}`);
                const newProductCategoryList = productCategories.filter((productCategory) => {
                    return productCategory.ProductCategoryID !== id;
                });
                setProductCategories(newProductCategoryList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!");
        }
    };

    const updateProductCategoryHandler = async (productCategory) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const productCategoryToUpdate = {
                    Id: productCategory.Id,
                    Name: productCategory.Name,
                    Description: productCategory.Description
                };
                console.log(productCategoryToUpdate);
                const response = await api.put("/productCategory", productCategory);
                const { productCategoryName } = response.data;
                setProductCategories(
                    productCategories.map((productCategory) => {
                        return productCategory.Name === productCategoryName ? { ...response.data } : productCategory;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            toast.error("Failed to update!");
        }
    };

    const onProductCategoryNameUpdate = (productCategory, event) => {
        const { value } = event.target;
        const data = [...rows];
        productCategory.Name = value;
        initRow(data);
        console.log(productCategory)
    };

    const onDescriptionUpdate = (productCategory, event) => {
        const { value } = event.target;
        const data = [...rows];
        productCategory.Description = value;
        initRow(data);
        console.log(productCategory)
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllProductCategories = async () => {
            const allProductCategories = await retrieveProductCategories();
            if (allProductCategories) setProductCategories(allProductCategories);
        };

        getAllProductCategories();
        setAdded(false);
        setDeleted(false);
        setUpdated(false);

    }, [added, deleted, updated]);

    const DisplayData = productCategories.map(
        (productCategory) => {
            return (
                <tr key={productCategory.Id}>
                    <td>
                        {productCategory.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            value={productCategory.Name}
                            onChange={(event) => onProductCategoryNameUpdate(productCategory, event)}
                            name="name"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={productCategory.Description}
                            onChange={(event) => onDescriptionUpdate(productCategory, event)}
                            name="description"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updateProductCategoryHandler(productCategory)}
                        >
                            Update
                        </button>
                        <button
                            className="buttonDelete"
                            onClick={() => removeProductCategoryHandler(productCategory.Id)}
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
            <AddProductCategoryModal addProductCategoryHandler={addProductCategoryHandler} open={open} handleOpen={handleOpen} />
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

export default ProductCategoryDataDisplay;