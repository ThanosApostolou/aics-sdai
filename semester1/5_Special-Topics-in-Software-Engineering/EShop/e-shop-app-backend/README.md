## Local Development
```
dotnet watch run
```

## DB

### Initialize
Run `sql/initialize-db.sql` with your mysql client

Run these commands
```
dotnet tool install --global dotnet-ef
cd EshopAPI
dotnet ef database update
```

### Update DB after entities changes
```
cd EshopAPI
dotnet ef migrations add Update1
dotnet ef database update
```