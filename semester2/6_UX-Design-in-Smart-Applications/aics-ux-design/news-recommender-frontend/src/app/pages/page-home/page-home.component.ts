import { Component, OnInit } from '@angular/core';
import { PagesEnum } from 'src/app/modules/core/enums/pages-enum';
import { GlobalStateStoreService } from 'src/app/modules/core/global-state-store.service';

@Component({
  selector: 'app-page-home',
  templateUrl: './page-home.component.html',
  styleUrls: ['./page-home.component.scss']
})
export class PageHomeComponent implements OnInit {
  constructor(private globalStateStoreService: GlobalStateStoreService) {
  }
  ngOnInit() {
    setTimeout(() => {
      this.globalStateStoreService.activePage.set(PagesEnum.HOME)
    })
  }
}
