import EshopUserDataDisplay from '../components/EshopUser/EshopUserTable'
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function EshopUser() {
  return (
    <div className="container">
      <h2>Our Users!!!</h2>
      <span><br /><br /></span>
      <Toaster/>
      <EshopUserDataDisplay />
      <div className="nav-bar-container-light">
        <BottomBar />
      </div>
    </div>
  );
}
export default EshopUser;