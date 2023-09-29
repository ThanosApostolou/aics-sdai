import { GlobalState } from "../../../modules/core/global-state";
import { FuzzyProfileDto } from '../../../modules/fuzzy/dtos/fuzzy-profile-dto';
import { FUZZY_CONSTANTS } from "../../../modules/fuzzy/fuzzy-constants";
import { CreateFuzzyProfileResponseDto } from './dtos/create-fuzzy-profile-dto';
import { FetchFuzzyProfilesResponseDto } from "./dtos/fetch-fuzzy-profiles-dto";

export class FuzzySettingsService {
    static async fetchAllFuzzyProfiles(): Promise<string[]> {
        return Promise.resolve([FUZZY_CONSTANTS.DEFAULT])
    }


    static async fetchFuzzyProfiles(): Promise<FetchFuzzyProfilesResponseDto> {
        const apiConsumer = GlobalState.instance.apiConsumer;
        const fetchFuzzyProfilesUrl = '/admin/fuzzy_settings/fetch_fuzzy_profiles'

        const response = await apiConsumer.get(fetchFuzzyProfilesUrl);
        const fetchProvidersListResponseDto: FetchFuzzyProfilesResponseDto = FetchFuzzyProfilesResponseDto.fromObj(response.data);
        return fetchProvidersListResponseDto;
    }

    static async createFuzzyProfile(fuzzyProfileDto: FuzzyProfileDto): Promise<CreateFuzzyProfileResponseDto> {
        const apiConsumer = GlobalState.instance.apiConsumer;
        const createFuzzyProfileUrl = '/admin/fuzzy_settings/create_fuzzy_profile'

        const response = await apiConsumer.post(createFuzzyProfileUrl, fuzzyProfileDto);
        const createFuzzyProfileResponseDto: CreateFuzzyProfileResponseDto = CreateFuzzyProfileResponseDto.fromObj(response.data);
        return createFuzzyProfileResponseDto;
    }

    static async updateFuzzyProfile(fuzzyProfileDto: FuzzyProfileDto): Promise<CreateFuzzyProfileResponseDto> {
        const apiConsumer = GlobalState.instance.apiConsumer;
        const updateFuzzyProfileUrl = '/admin/fuzzy_settings/update_fuzzy_profile'

        const response = await apiConsumer.put(updateFuzzyProfileUrl, fuzzyProfileDto);
        const createFuzzyProfileResponseDto: CreateFuzzyProfileResponseDto = CreateFuzzyProfileResponseDto.fromObj(response.data);
        return createFuzzyProfileResponseDto;
    }

    static async deleteFuzzyProfile(name: string): Promise<CreateFuzzyProfileResponseDto> {
        const apiConsumer = GlobalState.instance.apiConsumer;
        const deleteFuzzyProfileUrl = '/admin/fuzzy_settings/delete_fuzzy_profile'

        const response = await apiConsumer.delete(deleteFuzzyProfileUrl, {
            params: {
                name: name
            }
        });
        const createFuzzyProfileResponseDto: CreateFuzzyProfileResponseDto = CreateFuzzyProfileResponseDto.fromObj(response.data);
        return createFuzzyProfileResponseDto;
    }
}