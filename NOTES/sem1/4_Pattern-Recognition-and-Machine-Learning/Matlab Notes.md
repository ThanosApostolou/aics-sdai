
### [MATLAB Operators and Special Characters](https://www.mathworks.com/help/matlab/matlab_prog/matlab-operators-and-special-characters.html)

>[!Arithmetic Operators]+
| Symbol | Role                                              | More Information                                                        |
| ------ | ------------------------------------------------- | ----------------------------------------------------------------------- |
| `+`      | Addition                                          | [plus](https://www.mathworks.com/help/matlab/ref/plus.html)             |
| `+`      | Unary plus                                        | [uplus](https://www.mathworks.com/help/matlab/ref/uplus.html)           |
| `-`     | Subtraction                                       | [minus](https://www.mathworks.com/help/matlab/ref/minus.html)           |
| `-`     | Unary minus                                       | [uminus](https://www.mathworks.com/help/matlab/ref/uminus.html)         |
| `.*`    | Element-wise multiplication                       | [times](https://www.mathworks.com/help/matlab/ref/times.html)           |
| `*`     | Matrix multiplication                             | [mtimes](https://www.mathworks.com/help/matlab/ref/mtimes.html)         |
| `./`     | Element-wise right division                       | [rdivide](https://www.mathworks.com/help/matlab/ref/rdivide.html)       |
| `/`      | Matrix right division                             | [mrdivide](https://www.mathworks.com/help/matlab/ref/mrdivide.html)     |
| `.\`    | Element-wise left division                        | [ldivide](https://www.mathworks.com/help/matlab/ref/ldivide.html)       |
| `\`     | Matrix left division <br>(also known as backslash) | [mldivide](https://www.mathworks.com/help/matlab/ref/mldivide.html)     |
| `.^`     | Element-wise power                                | [power](https://www.mathworks.com/help/matlab/ref/power.html)           |
| `^`      | Matrix power                                      | [mpower](https://www.mathworks.com/help/matlab/ref/mpower.html)         |
| `.'`     | Transpose                                         | [transpose](https://www.mathworks.com/help/matlab/ref/transpose.html)   |
| `'`      | Complex conjugate transpose                       | [ctranspose](https://www.mathworks.com/help/matlab/ref/ctranspose.html) |

>[!Relational Operators]+
| Symbol | Role                     | More Information                                        |
| ------ | ------------------------ | ------------------------------------------------------- |
| `==`   | Equal to                 | [eq](https://www.mathworks.com/help/matlab/ref/eq.html) |
| `~=`   | Not equal to             | [ne](https://www.mathworks.com/help/matlab/ref/ne.html) |
| `>`    | Greater than             | [gt](https://www.mathworks.com/help/matlab/ref/gt.html) |
| `>=`   | Greater than or equal to | [ge](https://www.mathworks.com/help/matlab/ref/ge.html) |
| `<`    | Less than                | [lt](https://www.mathworks.com/help/matlab/ref/lt.html) |
| `<=`     | Less than or equal to    | [le](https://www.mathworks.com/help/matlab/ref/le.html) |

>[!Logical Operators]+
| Symbol | Role                                     | More Information                |                                                    |
| ------ | ---------------------------------------- | ----------------------------------------------------------------------------------- |
| `&`      | Find logical AND                         | [and](https://www.mathworks.com/help/matlab/ref/and.html)                           |
|  \|    | Find logical OR                          | [or](https://www.mathworks.com/help/matlab/ref/or.html)                             |
| `&&`     | Find logical AND (with short-circuiting) | [Short-Circuit AND](https://www.mathworks.com/help/matlab/ref/shortcircuitand.html) |
| \|\|     | Find logical OR (with short-circuiting)  | [Short-Circuit OR](https://www.mathworks.com/help/matlab/ref/shortcircuitor.html)   |
| `~`      | Find logical NOT                         | [not](https://www.mathworks.com/help/matlab/ref/not.html)                           |

### [Entering Commands](https://www.mathworks.com/help/matlab/entering-commands.html)

>[!Functions]+
| [ans](https://www.mathworks.com/help/matlab/ref/ans.html)                       | Most recent answer                        |
| ------------------------------------------------------------------------------- | ----------------------------------------- |
| [clc](https://www.mathworks.com/help/matlab/ref/clc.html)                       | Clear Command Window                      |
| [diary](https://www.mathworks.com/help/matlab/ref/diary.html)                   | Log Command Window text to file           |
| [format](https://www.mathworks.com/help/matlab/ref/format.html)                 | Set output display format                 |
| [home](https://www.mathworks.com/help/matlab/ref/home.html)                     | Send cursor home                          |
| [iskeyword](https://www.mathworks.com/help/matlab/ref/iskeyword.html)           | Determine whether input is MATLAB keyword |
| [more](https://www.mathworks.com/help/matlab/ref/more.html)                     | Control paged output in Command Window    |
| [commandwindow](https://www.mathworks.com/help/matlab/ref/commandwindow.html)   | Select the Command Window                 |
| [commandhistory](https://www.mathworks.com/help/matlab/ref/commandhistory.html) | Open Command History window               |

### [Matrices and Arrays](https://www.mathworks.com/help/matlab/matrices-and-arrays.html?s_tid=CRUX_lftnav)

>[!Create and Combine Arrays]+
| [zeros](https://www.mathworks.com/help/matlab/ref/zeros.html)            | Create array of all zeros                                 |
| ------------------------------------------------------------------------ | --------------------------------------------------------- |
| [ones](https://www.mathworks.com/help/matlab/ref/ones.html)              | Create array of all ones                                  |
| [rand](https://www.mathworks.com/help/matlab/ref/rand.html)              | Uniformly distributed random numbers                      |
| [TRUE](https://www.mathworks.com/help/matlab/ref/true.html)              | Logical 1 (true)                                          |
| [FALSE](https://www.mathworks.com/help/matlab/ref/false.html)            | Logical 0 (false)                                         |
| [eye](https://www.mathworks.com/help/matlab/ref/eye.html)                | Identity matrix                                           |
| [diag](https://www.mathworks.com/help/matlab/ref/diag.html)              | Create diagonal matrix or get diagonal elements of matrix |
| [blkdiag](https://www.mathworks.com/help/matlab/ref/blkdiag.html)        | Block diagonal matrix                                     |
| [cat](https://www.mathworks.com/help/matlab/ref/double.cat.html)         | Concatenate arrays                                        |
| [horzcat](https://www.mathworks.com/help/matlab/ref/double.horzcat.html) | Concatenate arrays horizontally                           |
| [vertcat](https://www.mathworks.com/help/matlab/ref/double.vertcat.html) | Concatenate arrays vertically                             |
| [repelem](https://www.mathworks.com/help/matlab/ref/repelem.html)        | Repeat copies of array elements                           |
| [repmat](https://www.mathworks.com/help/matlab/ref/repmat.html)          | Repeat copies of array                                    |

>[!Create Grids]+
| [linspace](https://www.mathworks.com/help/matlab/ref/linspace.html)   | Generate linearly spaced vector          |
| --------------------------------------------------------------------- | ---------------------------------------- |
| [logspace](https://www.mathworks.com/help/matlab/ref/logspace.html)   | Generate logarithmically spaced vector   |
| [freqspace](https://www.mathworks.com/help/matlab/ref/freqspace.html) | Frequency spacing for frequency response |
| [meshgrid](https://www.mathworks.com/help/matlab/ref/meshgrid.html)   | 2-D and 3-D grids                        |
| [ndgrid](https://www.mathworks.com/help/matlab/ref/ndgrid.html)       | Rectangular grid in N-D space            |

>[!Determine Size, Shape, and Order]+
| [length](https://www.mathworks.com/help/matlab/ref/length.html)             | Length of largest array dimension            |
| --------------------------------------------------------------------------- | -------------------------------------------- |
| [size](https://www.mathworks.com/help/matlab/ref/size.html)                 | Array size                                   |
| [ndims](https://www.mathworks.com/help/matlab/ref/double.ndims.html)        | Number of array dimensions                   |
| [numel](https://www.mathworks.com/help/matlab/ref/double.numel.html)        | Number of array elements                     |
| [isscalar](https://www.mathworks.com/help/matlab/ref/isscalar.html)         | Determine whether input is scalar            |
| [isvector](https://www.mathworks.com/help/matlab/ref/isvector.html)         | Determine whether input is vector            |
| [ismatrix](https://www.mathworks.com/help/matlab/ref/ismatrix.html)         | Determine whether input is matrix            |
| [isrow](https://www.mathworks.com/help/matlab/ref/isrow.html)               | Determine if input is row vector             |
| [iscolumn](https://www.mathworks.com/help/matlab/ref/iscolumn.html)         | Determine if input is column vector          |
| [isempty](https://www.mathworks.com/help/matlab/ref/double.isempty.html)    | Determine whether array is empty             |
| [issorted](https://www.mathworks.com/help/matlab/ref/issorted.html)         | Determine if array is sorted                 |
| [issortedrows](https://www.mathworks.com/help/matlab/ref/issortedrows.html) | Determine if matrix or table rows are sorted |
| [isuniform](https://www.mathworks.com/help/matlab/ref/isuniform.html)       | Determine if vector is uniformly spaced      |

>[!Reshape and Rearrange]+
| [sort](https://www.mathworks.com/help/matlab/ref/sort.html)                | Sort array elements              |
| -------------------------------------------------------------------------- | -------------------------------- 
| [sortrows](https://www.mathworks.com/help/matlab/ref/double.sortrows.html) | Sort rows of matrix or table     |
| [flip](https://www.mathworks.com/help/matlab/ref/flip.html)                | Flip order of elements           |
| [fliplr](https://www.mathworks.com/help/matlab/ref/fliplr.html)            | Flip array left to right         |
| [flipud](https://www.mathworks.com/help/matlab/ref/flipud.html)            | Flip array up to down            |
| [rot90](https://www.mathworks.com/help/matlab/ref/rot90.html)              | Rotate array 90 degrees          |
| [transpose](https://www.mathworks.com/help/matlab/ref/transpose.html)      | Transpose vector or matrix       |
| [ctranspose](https://www.mathworks.com/help/matlab/ref/ctranspose.html)    | Complex conjugate transpose      |
| [permute](https://www.mathworks.com/help/matlab/ref/permute.html)          | Permute array dimensions         |
| [ipermute](https://www.mathworks.com/help/matlab/ref/ipermute.html)        | Inverse permute array dimensions |
| [circshift](https://www.mathworks.com/help/matlab/ref/circshift.html)      | Shift array circularly           |
| [shiftdim](https://www.mathworks.com/help/matlab/ref/shiftdim.html)        | Shift array dimensions           |
| [reshape](https://www.mathworks.com/help/matlab/ref/reshape.html)          | Reshape array                    |
| [squeeze](https://www.mathworks.com/help/matlab/ref/squeeze.html)          | Remove dimensions of length 1    |

>[!Indexing]+
| [colon](https://www.mathworks.com/help/matlab/ref/colon.html)     | Vector creation, array subscripting, and for\-loop iteration |
| ----------------------------------------------------------------- | ------------------------------------------------------------ |
| [end](https://www.mathworks.com/help/matlab/ref/end.html)         | Terminate block of code or indicate last array index         |
| [ind2sub](https://www.mathworks.com/help/matlab/ref/ind2sub.html) | Convert linear indices to subscripts                         |
| [sub2ind](https://www.mathworks.com/help/matlab/ref/sub2ind.html) | Convert subscripts to linear indices                         |

| command                       | description         |
| ----------------------------- | ------------------- |
| `whos`                        |                     |
| `size(M)`                     |                     |
| `clc`, `clear M`, `clear all` |                     |
| `A.'` or `transpose(A)`        | Transposes Matrix A |
| `%` or `...`| Comment             |
| `;`                           | Supresses Output    |
