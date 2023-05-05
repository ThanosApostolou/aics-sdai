import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import "../../App.css";
import AddProductModal from "./AddProductModal";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";
import Select from 'react-select';

function ProductDataDisplay() {
    const [ProductCategories, setProductCategories] = useState([]);
    const [selectedProductCategory, setselectedProductCategory] = useState("");
    const [products, setProducts] = useState([]);
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

    const handleChangeProductCategory = (selectedProductCategory) => {
        setselectedProductCategory(selectedProductCategory);
    };

    const retrieveProducts = async () => {
        const response = await api.get("/product");
        return response.data;
    }

    const retrieveProductCategories = async () => {
        const response = await api.get("/productCategory");
        return response.data;
    }

    async function addProductHandler(product) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = { ...product }
                const response = await api.post("/product", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            toast.error("Failed to Add!")
        }
    };

    const removeProductHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/product/${id}`);
                const newProductList = products.filter((product) => {
                    return product.ProductID !== id;
                });
                setProducts(newProductList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!")
        }
    };

    const updateProductHandler = async (product) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const productToUpdate = {
                    Id: product.Id,
                    Name: product.Name,
                    Description: product.Description,
                    Image: product.Image,
                    Availability: product.Availability,
                    ProductCategoryId: selectedProductCategory ? selectedProductCategory.value.Id : product.ProductCategoryId,
                    ProductCategory: selectedProductCategory ?
                        { Id: selectedProductCategory.value.Id, Name: selectedProductCategory.value.Name, Description: selectedProductCategory.value.Description }
                        :
                        { Id: product.ProductCategoryId, Name: "", Description: "" }
                };
                await api.put("/product", productToUpdate);
                setProducts(
                    products.map((existingProduct) => {
                        return existingProduct.Id === productToUpdate.Id
                            ? { ...productToUpdate }
                            : existingProduct;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            toast.error("Failed to update!");
        }
    };

    const onProductNameUpdate = (product, event) => {
        const { value } = event.target;
        const data = [...rows];
        product.Name = value;
        initRow(data);
    };

    const onDescriptionUpdate = (product, event) => {
        const { value } = event.target;
        const data = [...rows];
        product.Description = value;
        initRow(data);
    };

    const onImageUpdate = (product, event) => {
        const { value } = event.target;
        const data = [...rows];
        product.Image = value;
        initRow(data);
    };

    const onAvailabilityUpdate = (product, event) => {
        const { value } = event.target;
        const data = [...rows];
        product.Availability = value;
        initRow(data);
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllProducts = async () => {
            const allProducts = await retrieveProducts();
            if (allProducts) setProducts(allProducts);
        };

        const getAllProductCategories = async () => {
            const allProductCategories = await retrieveProductCategories();

            if (allProductCategories) setProductCategories(
                allProductCategories.map((ProductCategoryNavigation) => {
                    return {
                        label: ProductCategoryNavigation.Name,
                        value: ProductCategoryNavigation
                    }
                })
            );

        };


        getAllProducts();
        getAllProductCategories();
        setDeleted(false);
        setUpdated(false);
        setAdded(false);


    }, [added, deleted, updated]);

    const DisplayData = products.map(
        (product) => {
            return (
                <tr key={product.Id}>
                    <td>
                        {product.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            value={product.Name}
                            onChange={(event) => onProductNameUpdate(product, event)}
                            name="productName"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={product.Description}
                            onChange={(event) => onDescriptionUpdate(product, event)}
                            name="description"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={product.Image}
                            onChange={(event) => onImageUpdate(product, event)}
                            name="image"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={product.Availability}
                            onChange={(event) => onAvailabilityUpdate(product, event)}
                            name="availability"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={product.ProductCategory.Name}
                            name="productCategoryID"
                            className="form-control"
                        />
                        <Select
                            value={selectedProductCategory}
                            onChange={handleChangeProductCategory}
                            options={ProductCategories}
                        />
                    </td>
                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updateProductHandler(product)}
                        >
                            Update
                        </button>
                        <span> </span>
                        <button
                            className="buttonDelete"
                            onClick={() => removeProductHandler(product.Id)}
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
            <AddProductModal addProductHandler={addProductHandler} open={open} handleOpen={handleOpen} />
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Shop Name</th>
                        <th>Description</th>
                        <th>Image</th>
                        <th>Availability</th>
                        <th>Category</th>
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

export default ProductDataDisplay;