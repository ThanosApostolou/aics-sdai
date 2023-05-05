import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import "../../App.css";
import AddShopModal from "./AddShopModal";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";
import Select from 'react-select';

function ShopDataDisplay() {
    const [ShopCategories, setShopCategories] = useState([]);
    const [selectedShopCategory, setSelectedShopCategory] = useState("");
    const [Sellers, setSellers] = useState([]);
    const [selectedSeller, setSelectedSeller] = useState("");
    const [shops, setShops] = useState([]);
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

    const handleChangeShopCategory = (selectedShopCategory) => {
        setSelectedShopCategory(selectedShopCategory);
    };

    const handleChangeSeller = (selectedSeller) => {
        setSelectedSeller(selectedSeller);
    };

    const retrieveShops = async () => {
        const response = await api.get("/shop");
        return response.data;
    }

    const retrieveShopCategories = async () => {
        const response = await api.get("/shopCategory");
        return response.data;
    }

    const retrieveSellers = async () => {
        const response = await api.get("/eshopUser");
        return response.data;
    }

    async function addShopHandler(shop) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = { ...shop }
                const response = await api.post("/shop", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            toast.error("Failed to Add!")
        }
    };

    const removeShopHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/shop/${id}`);
                const newShopList = shops.filter((shop) => {
                    return shop.ShopID !== id;
                });
                setShops(newShopList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!")
        }
    };

    const updateShopHandler = async (shop) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const shopToUpdate = {
                    Id: shop.Id,
                    Seller: selectedSeller ? selectedSeller.value.Id : shop.Seller,
                    ShopCategory: selectedShopCategory ? selectedShopCategory.value.Id : shop.ShopCategory,
                    Name: shop.Name,
                    Image: shop.Image,
                    Description: shop.Description,
                    ShopCategoryNavigation: selectedShopCategory ?
                        { Id: selectedShopCategory.value.Id, Name: selectedShopCategory.value.Name, Description: selectedShopCategory.value.Description }
                        :
                        { Id: shop.ShopCategory, Name: "", Description: "" },
                    SellerNavigation: selectedSeller ?
                        { Id: selectedSeller.value.Id, Username: selectedSeller.value.Username, Email: selectedSeller.value.Email, Address: selectedSeller.value.Address}
                        :
                        { Id: shop.Seller, Username: "", Email: "", Address: "" }
                };
                console.log(shopToUpdate);
                await api.put("/shop", shopToUpdate);
                setShops(
                    shops.map((existingShop) => {
                        return existingShop.ShopCategory === shopToUpdate.ShopCategory
                            ? { ...shopToUpdate }
                            : existingShop;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            toast.error("Failed to update!");
        }
    };

    const onShopNameUpdate = (shop, event) => {
        const { value } = event.target;
        const data = [...rows];
        shop.Name = value;
        initRow(data);
    };

    const onDescriptionUpdate = (shop, event) => {
        const { value } = event.target;
        const data = [...rows];
        shop.Description = value;
        initRow(data);
    };

    const onImageUpdate = (shop, event) => {
        const { value } = event.target;
        const data = [...rows];
        shop.Image = value;
        initRow(data);
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllShops = async () => {
            const allShops = await retrieveShops();
            if (allShops) setShops(allShops);
        };

        const getAllShopCategories = async () => {
            const allShopCategories = await retrieveShopCategories();

            if (allShopCategories) setShopCategories(
                allShopCategories.map((ShopCategoryNavigation) => {
                    return {
                        label: ShopCategoryNavigation.Name,
                        value: ShopCategoryNavigation
                    }
                })
            );

        };

        const getAllSellers = async () => {
            const allSellers = await retrieveSellers();

            if (allSellers) setSellers(
                allSellers.map((SellerNavigation) => {
                    return {
                        label: SellerNavigation.Username,
                        value: SellerNavigation
                    }
                })
            );
        };

        getAllShops();
        getAllShopCategories();
        getAllSellers();
        setDeleted(false);
        setUpdated(false);
        setAdded(false);


    }, [added, deleted, updated]);

    const DisplayData = shops.map(
        (shop) => {
            return (
                <tr key={shop.Id}>
                    <td>
                        {shop.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            value={shop.Name}
                            onChange={(event) => onShopNameUpdate(shop, event)}
                            name="shopName"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={shop.ShopCategoryNavigation.Name}
                            name="shopCategoryID"
                            className="form-control"
                        />
                        <Select
                            value={selectedShopCategory}
                            onChange={handleChangeShopCategory}
                            options={ShopCategories}
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={shop.Image}
                            onChange={(event) => onImageUpdate(shop, event)}
                            name="image"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={shop.Description}
                            onChange={(event) => onDescriptionUpdate(shop, event)}
                            name="description"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={shop.SellerNavigation.Username}
                            name="SellerNavigationID"
                            className="form-control"
                        />
                        <Select
                            value={selectedSeller}
                            onChange={handleChangeSeller}
                            options={Sellers}
                        />
                    </td>
                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updateShopHandler(shop)}
                        >
                            Update
                        </button>
                        <span> </span>
                        <button
                            className="buttonDelete"
                            onClick={() => removeShopHandler(shop.Id)}
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
            <AddShopModal addShopHandler={addShopHandler} open={open} handleOpen={handleOpen} />
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Shop Name</th>
                        <th>Category</th>
                        <th>Image</th>
                        <th>Description</th>
                        <th>Seller</th>
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

export default ShopDataDisplay;