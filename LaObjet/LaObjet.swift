//
//  LaObjet.swift
//  macOS
//
//  Created by Kota Nakano on 2017/02/08.
//
//

import Accelerate

public typealias LaObjet = la_object_t

private let attr: la_attribute_t = la_attribute_t(LA_ATTRIBUTE_ENABLE_LOGGING)
private let hint: la_hint_t = la_hint_t(LA_NO_HINT)

extension LaObjet {
	final public var status: Int {
		return la_status(self)
	}
	final public var T: LaObjet {
		return la_transpose(self)
	}
	final public var count: UInt {
		return la_vector_length(self)
	}
	final public var rows: UInt {
		return la_matrix_rows(self)
	}
	final public var cols: UInt {
		return la_matrix_cols(self)
	}
	final public var L1: Float {
		return la_norm_as_float(self, la_norm_t(LA_L1_NORM))
	}
	final public var L2: Float {
		return la_norm_as_float(self, la_norm_t(LA_L2_NORM))
	}
	final public var LINF: Float {
		return la_norm_as_float(self, la_norm_t(LA_LINF_NORM))
	}
	final public subscript(index: Int) -> LaObjet {
		return la_splat_from_vector_element(self, la_index_t(index))
	}
	final public subscript(rows: Int, cols: Int) -> LaObjet {
		return la_splat_from_matrix_element(self, la_index_t(rows), la_index_t(cols))
	}
	final public subscript(range: Range<Int>) -> LaObjet {
		return la_vector_slice(self, range.lowerBound, 1, la_count_t(range.count))
	}
	final public subscript(rows: Range<Int>, cols: Range<Int>) -> LaObjet {
		return la_matrix_slice(self, rows.lowerBound, cols.lowerBound, 1, 1, la_count_t(rows.count), la_count_t(cols.count))
	}
	final public func fill(rows: Int, cols: Int) -> LaObjet {
		return la_matrix_from_splat(self, la_count_t(rows), la_count_t(cols))
	}
	final public func render(to: UnsafeRawPointer, stride: Int? = nil) {
		la_matrix_to_float_buffer(UnsafeMutablePointer<Float>(OpaquePointer(to)), la_count_t(stride ?? Int(la_matrix_cols(self))), self)
	}
}
public prefix func -(_ lhs: LaObjet) -> LaObjet {
	return lhs.count == 0 ? la_scale_with_float(lhs, -1) : la_scale_with_float(lhs, -1)
}

// MARK: - Add
public func +(_ lhs: LaObjet, _ rhs: LaObjet) -> LaObjet {
	return la_sum(lhs, rhs)
}
public func +(_ lhs: LaObjet, _ rhs: Float) -> LaObjet {
	return la_sum(lhs, la_splat_from_float(rhs, attr))
}
public func +(_ lhs: Float, _ rhs: LaObjet) -> LaObjet {
	return la_sum(la_splat_from_float(lhs, attr), rhs)
}

// MARK: - Subtract
public func -(_ lhs: LaObjet, _ rhs: LaObjet) -> LaObjet {
	return la_difference(lhs, rhs)
}
public func -(_ lhs: LaObjet, _ rhs: Float) -> LaObjet {
	return la_difference(lhs, la_splat_from_float(rhs, attr))
}
public func -(_ lhs: Float, _ rhs: LaObjet) -> LaObjet {
	return la_difference(la_splat_from_float(lhs, attr), rhs)
}

// MARK: - Multiplication
public func *(_ lhs: LaObjet, _ rhs: LaObjet) -> LaObjet {
	return la_elementwise_product(lhs, rhs)
}
public func *(_ lhs: Float, _ rhs: LaObjet) -> LaObjet {
	return la_scale_with_float(rhs, lhs)
}
public func *(_ lhs: LaObjet, _ rhs: Float) -> LaObjet {
	return la_scale_with_float(lhs, rhs)
}

//2 ops
public func inner_product(_ lhs: LaObjet, _ rhs: LaObjet) -> LaObjet {
	return la_inner_product(lhs, rhs)
}
public func outer_product(_ lhs: LaObjet, _ rhs: LaObjet) -> LaObjet {
	return la_outer_product(lhs, rhs)
}
public func matrix_product(_ lhs: LaObjet, _ rhs: LaObjet) -> LaObjet {
	return la_matrix_product(lhs, rhs)
}
public func matrix_solve(_ lhs: LaObjet, _ rhs: LaObjet) -> LaObjet {
	return la_solve(lhs, rhs)
}

public func make(eye: Int) -> LaObjet {
	return la_identity_matrix(la_count_t(eye), la_scalar_type_t(LA_SCALAR_TYPE_FLOAT), attr)
}
public func make(eye: LaObjet) -> LaObjet {
	return la_diagonal_matrix_from_vector(eye, 0)
}
public func make(value: Float) -> LaObjet {
	return la_splat_from_float(value, attr)
}
public func make(array: Array<Float>, rows: Int, cols: Int) -> LaObjet {
	return la_matrix_from_float_buffer(array, la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, attr)
}
public func make(copy pointer: UnsafeRawPointer, rows: Int, cols: Int) -> LaObjet {
	return la_matrix_from_float_buffer(UnsafeMutablePointer<Float>(OpaquePointer(pointer)), la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, attr)
	
}
public func make(nocopy pointer: UnsafeRawPointer, rows: Int, cols: Int) -> LaObjet {
	return la_matrix_from_float_buffer_nocopy(UnsafeMutablePointer<Float>(OpaquePointer(pointer)), la_count_t(rows), la_count_t(cols), la_count_t(cols), hint, nil, attr)

}
