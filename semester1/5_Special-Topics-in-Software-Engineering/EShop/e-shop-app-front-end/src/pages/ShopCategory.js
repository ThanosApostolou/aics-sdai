import ShopCategoryDataDisplay from '../components/ShopCategory/ShopCategoryTable'
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function ShopCategory() {
  return (
    <div className="container">
      <h2>Our Shop Categories!!!</h2>
      <span><br /><br /></span>
      <Toaster/>
      <ShopCategoryDataDisplay />
      <div className="nav-bar-container-light">
        <BottomBar />
      </div>
    </div>
  );
}
export default ShopCategory;