#include "system_info.h"

#include <Windows.h>
#include <Wbemidl.h>
#include <wrl/client.h>
#include <comdef.h>

#pragma comment(lib, "wbemuuid.lib")
using Microsoft::WRL::ComPtr;

class ComInitializer
{
public:
	inline explicit ComInitializer(DWORD coinit = COINIT_MULTITHREADED) noexcept
		: _initialized{SUCCEEDED(::CoInitializeEx(nullptr, coinit))} {}

	inline ~ComInitializer() { if (_initialized) ::CoUninitialize(); }

	[[nodiscard]] inline bool isInitialized() const noexcept { return _initialized; }

private:
	bool _initialized = false;
};


std::vector<RamInfo> queryMemoryInfo() noexcept try
{
	ComInitializer initializer;
	if (!initializer.isInitialized())
		return {};

	HRESULT hr = CoInitializeSecurity(nullptr, -1, nullptr, nullptr,
		RPC_C_AUTHN_LEVEL_DEFAULT, RPC_C_IMP_LEVEL_IMPERSONATE,
		nullptr, EOAC_NONE, nullptr);
	if (FAILED(hr)) return {};

	ComPtr<IWbemLocator> locator;
	hr = CoCreateInstance(CLSID_WbemLocator, nullptr, CLSCTX_INPROC_SERVER,
		IID_PPV_ARGS(&locator));
	if (FAILED(hr)) return {};

	ComPtr<IWbemServices> services;
	hr = locator->ConnectServer(
		BSTR(L"ROOT\\CIMV2"),
		nullptr, nullptr, nullptr, 0, nullptr, nullptr,
		&services);
	if (FAILED(hr)) return {};

	hr = CoSetProxyBlanket(services.Get(),
		RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr,
		RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE,
		nullptr, EOAC_NONE);
	if (FAILED(hr)) return {};

	ComPtr<IEnumWbemClassObject> enumerator;
	hr = services->ExecQuery(
		BSTR(L"WQL"),
		BSTR(L"SELECT * FROM Win32_PhysicalMemory"),
		WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
		nullptr, &enumerator);
	if (FAILED(hr)) return {};

	std::vector<RamInfo> ramInfo;
	for (;;)
	{
		ComPtr<IWbemClassObject> obj;
		ULONG returned = 0;
		if (enumerator->Next(WBEM_INFINITE, 1, &obj, &returned) != S_OK)
			break;

		RamInfo info;
		_variant_t vt;
		if (SUCCEEDED(obj->Get(L"SMBIOSMemoryType", 0, &vt, nullptr, nullptr)))
		{
			switch (vt.uintVal)
			{
			case 20: info.ddrStandardNumber = 1; break;
			case 21: info.ddrStandardNumber = 2; break;
			case 24: info.ddrStandardNumber = 3; break;
			case 26: info.ddrStandardNumber = 4; break;
			case 27: info.ddrStandardNumber = 5; break;
			}
		}
		if (SUCCEEDED(obj->Get(L"Speed", 0, &vt, nullptr, nullptr)))
		{
			info.moduleMaxSpeed = vt.uintVal;
		}
		if (SUCCEEDED(obj->Get(L"ConfiguredClockSpeed", 0, &vt, nullptr, nullptr)))
		{
			info.clock = vt.uintVal;
		}
		if (SUCCEEDED(obj->Get(L"DeviceLocator", 0, &vt, nullptr, nullptr)))
		{
			info.bank = _com_util::ConvertBSTRToString(vt.bstrVal);
		}
		if (SUCCEEDED(obj->Get(L"Capacity", 0, &vt, nullptr, nullptr)))
		{
			info.capacity = _wtoi64(vt.bstrVal);
		}
		if (SUCCEEDED(obj->Get(L"Manufacturer", 0, &vt, nullptr, nullptr)))
		{
			info.manufacturer = _com_util::ConvertBSTRToString(vt.bstrVal);
		}
		if (SUCCEEDED(obj->Get(L"PartNumber", 0, &vt, nullptr, nullptr)))
		{
			info.model = _com_util::ConvertBSTRToString(vt.bstrVal);
		}

		obj->Get(L"FormFactor", 0, &vt, nullptr, nullptr);

		ramInfo.push_back(std::move(info));
	}

	return ramInfo;
} catch (...) {
	return {};
}
